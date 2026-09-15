from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "full"
STATIC = ROOT / "webapp" / "static"
RUNTIME = ROOT / "data" / "runtime"

REQUIRED_DATA = ("connectome_csr.npz", "node_metadata.npz", "stats.json")
missing_data = [name for name in REQUIRED_DATA if not (DATA / name).is_file()]
if missing_data:
    raise RuntimeError(
        "Full MaleCNS artifacts are missing: " + ", ".join(missing_data)
        + ". Download the official MaleCNS files into data/raw and run "
        + "python scripts/prepare_full_data.py. See data/README.md."
    )

print("Loading the complete 166,700-neuron MaleCNS graph...")
GRAPH = sparse.load_npz(DATA / "connectome_csr.npz").astype(np.float32)
META = np.load(DATA / "node_metadata.npz", allow_pickle=False)
STATS = json.loads((DATA / "stats.json").read_text(encoding="utf-8"))
POSITIONS = META["soma"].astype("<f4")
REGION_CODE = META["region_code"]
POSITION_BYTES = POSITIONS.tobytes() + REGION_CODE.tobytes()

MAX_WS_CONNECTIONS = int(os.getenv("MAX_WS_CONNECTIONS", "12"))
MAX_WS_PER_IP = int(os.getenv("MAX_WS_PER_IP", "2"))
MAX_WS_MESSAGES_PER_SECOND = int(os.getenv("MAX_WS_MESSAGES_PER_SECOND", "24"))
MAX_WS_MESSAGE_BYTES = 8 * 1024


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class WebSocketGate:
    """Small in-process guard for an intentionally public, compute-heavy demo."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.total = 0
        self.by_ip: dict[str, int] = {}

    async def acquire(self, ip: str) -> bool:
        async with self.lock:
            if self.total >= MAX_WS_CONNECTIONS or self.by_ip.get(ip, 0) >= MAX_WS_PER_IP:
                return False
            self.total += 1
            self.by_ip[ip] = self.by_ip.get(ip, 0) + 1
            return True

    async def release(self, ip: str) -> None:
        async with self.lock:
            self.total = max(0, self.total - 1)
            remaining = self.by_ip.get(ip, 1) - 1
            if remaining > 0:
                self.by_ip[ip] = remaining
            else:
                self.by_ip.pop(ip, None)


WS_GATE = WebSocketGate()


class AnonymousAnalytics:
    """Stores only hashes of browser-generated random IDs; never IP addresses."""

    def __init__(self) -> None:
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(RUNTIME / "analytics.sqlite", check_same_thread=False)
        self.lock = threading.Lock()
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS visitors (
                    visitor_hash TEXT PRIMARY KEY,
                    first_seen INTEGER NOT NULL,
                    last_seen INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_hash TEXT PRIMARY KEY,
                    visitor_hash TEXT NOT NULL,
                    first_seen INTEGER NOT NULL,
                    last_seen INTEGER NOT NULL
                );
            """)

    @staticmethod
    def _hash(value: str) -> str:
        import hashlib
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _valid(value: object) -> bool:
        return isinstance(value, str) and 16 <= len(value) <= 128 and value.replace("-", "").isalnum()

    def heartbeat(self, visitor_id: object, session_id: object) -> dict:
        if not self._valid(visitor_id) or not self._valid(session_id):
            raise ValueError("Invalid anonymous session identifier")
        now = int(time.time())
        visitor_hash, session_hash = self._hash(visitor_id), self._hash(session_id)
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO visitors(visitor_hash, first_seen, last_seen) VALUES (?, ?, ?) "
                "ON CONFLICT(visitor_hash) DO UPDATE SET last_seen=excluded.last_seen",
                (visitor_hash, now, now),
            )
            self.connection.execute(
                "INSERT INTO sessions(session_hash, visitor_hash, first_seen, last_seen) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_hash) DO UPDATE SET last_seen=excluded.last_seen",
                (session_hash, visitor_hash, now, now),
            )
            # Session hashes are anonymous and only retained for a short operational window.
            self.connection.execute("DELETE FROM sessions WHERE last_seen < ?", (now - 30 * 24 * 3600,))
            return self._summary(now)

    def _summary(self, now: int) -> dict:
        active = self.connection.execute(
            "SELECT COUNT(*) FROM sessions WHERE last_seen >= ?", (now - 90,)
        ).fetchone()[0]
        visitors = self.connection.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
        return {"active_sessions": int(active), "cumulative_visitors": int(visitors), "privacy": "anonymous_id_hashes_only"}


ANALYTICS = AnonymousAnalytics()


class BrainSession:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.n = GRAPH.shape[0]
        self.state = np.zeros(self.n, dtype=np.float32)
        self.tick = 0
        self.running = False
        self.stimulus = "visual"
        self.intensity = 0.0
        self.remaining = 0
        self.stim_indices = {
            item["key"]: np.flatnonzero((META["stim_code"] & (1 << item["bit"])) != 0)
            for item in STATS["stimuli"]
        }
        self.region_indices = [np.flatnonzero(REGION_CODE == i) for i in range(len(STATS["regions"]))]
        visual = self.stim_indices["visual"]
        visual_order = visual[np.argsort(POSITIONS[visual, 0] + .35 * POSITIONS[visual, 2])]
        self.game_sensors = np.array_split(visual_order, 4)
        self.readout = np.zeros((4, self.n), dtype=np.float32)
        self.game_running = False
        self.learning = True
        self.note_lane = int(self.rng.integers(0, 4))
        self.note_ticks = 14
        self.action = -1
        self.score = 0
        self.combo = 0
        self.hits = 0
        self.misses = 0
        self.reward = 0.0
        self.loss_history = deque(maxlen=80)
        self.activity_history = deque(maxlen=160)
        self.arena_running = False
        self.arena_learning = True
        self.arena_x = 0.22
        self.arena_y = 0.72
        self.arena_heading = -0.35
        self.arena_speed = 0.0
        self.arena_turn = 0.0
        self.arena_reward = 0.0
        self.arena_total_reward = 0.0
        self.arena_foods = 0
        self.arena_collisions = 0
        self.odor_memory = 0.18
        self.threat_ticks = 0
        self.food_x, self.food_y = 0.78, 0.27
        self.light_x, self.light_y = 0.22, 0.18
        self.obstacle_x, self.obstacle_y, self.obstacle_r = 0.53, 0.54, 0.105
        self.arena_sensors = {"odor": 0.0, "visual": 0.0, "touch": 0.0, "taste": 0.0,
                              "temperature": 0.0, "humidity": 0.0, "wind": 0.0}
        self.motor_indices = {
            "forward": np.flatnonzero(META["types"] == "DNp09"),
            "backward": np.flatnonzero(META["types"] == "MDN"),
            "steering": np.flatnonzero(np.isin(META["types"], ["DNa01", "DNa02"])),
        }
        steering = self.motor_indices["steering"]
        self.steering_left = steering[POSITIONS[steering, 0] < 0]
        self.steering_right = steering[POSITIONS[steering, 0] >= 0]
        visual_all = self.stim_indices["visual"]
        self.visual_left = visual_all[POSITIONS[visual_all, 0] < 0]
        self.visual_right = visual_all[POSITIONS[visual_all, 0] >= 0]
        olfactory_all = self.stim_indices["olfactory"]
        self.olfactory_left = olfactory_all[POSITIONS[olfactory_all, 0] < 0]
        self.olfactory_right = olfactory_all[POSITIONS[olfactory_all, 0] >= 0]
        self.physics_observation: dict | None = None
        self.physics_command = {"gain_left": 0.0, "gain_right": 0.0, "confidence": 0.0}
        self.physics_last_distance: float | None = None
        self.physics_gate = 0

    def reset_brain(self) -> None:
        self.state.fill(0)
        self.tick = 0
        self.remaining = 0
        self.activity_history.clear()

    def reset_learning(self) -> None:
        self.readout.fill(0)
        self.score = self.combo = self.hits = self.misses = 0
        self.loss_history.clear()

    def reset_arena(self, keep_learning: bool = True) -> None:
        self.arena_x, self.arena_y = 0.22, 0.72
        self.arena_heading = float(self.rng.uniform(-np.pi, np.pi))
        self.arena_speed = self.arena_turn = 0.0
        self.arena_reward = self.arena_total_reward = 0.0
        self.arena_foods = self.arena_collisions = 0
        self.threat_ticks = 0
        self.physics_observation = None
        self.physics_command = {"gain_left": 0.0, "gain_right": 0.0, "confidence": 0.0}
        self.physics_last_distance = None
        self.physics_gate = 0
        if not keep_learning:
            self.odor_memory = 0.18

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return float((value + np.pi) % (2 * np.pi) - np.pi)

    def _target_signal(self, tx: float, ty: float) -> tuple[float, float, float]:
        dx, dy = tx - self.arena_x, ty - self.arena_y
        distance = float(np.hypot(dx, dy))
        bearing = self._wrap_angle(float(np.arctan2(dy, dx)) - self.arena_heading)
        return distance, bearing, float(np.sin(bearing))

    def _arena_input(self, drive: np.ndarray) -> dict:
        if not self.arena_running:
            return {"food_distance": self._target_signal(self.food_x, self.food_y)[0], "food_bearing": 0.0,
                    "light_bearing": 0.0, "touch": 0.0, "odor": 0.0, "visual": 0.0, "taste": 0.0}
        if self.physics_observation is not None:
            obs = self.physics_observation
            odor = float(np.clip(obs.get("odor", 0.0), 0, 1))
            visual = float(np.clip(obs.get("visual", 0.0), 0, 1))
            touch = float(np.clip(obs.get("touch", 0.0), 0, 1))
            taste = float(np.clip(obs.get("taste", 0.0), 0, 1))
            temperature = float(np.clip(obs.get("temperature", 0.0), 0, 1))
            humidity = float(np.clip(obs.get("humidity", 0.0), 0, 1))
            wind = float(np.clip(obs.get("wind", 0.0), 0, 1))
            bearing = float(np.clip(obs.get("target_bearing", 0.0), -np.pi, np.pi))
            gradient = float(np.sin(bearing))
            drive[self.olfactory_left] += np.float32(odor * (0.16 + 0.08 * max(0, gradient)))
            drive[self.olfactory_right] += np.float32(odor * (0.16 + 0.08 * max(0, -gradient)))
            drive[self.visual_left] += np.float32(visual * (0.11 + 0.08 * max(0, gradient)))
            drive[self.visual_right] += np.float32(visual * (0.11 + 0.08 * max(0, -gradient)))
            if touch:
                drive[self.stim_indices["touch"]] += np.float32(0.38 * touch)
            proprioception = float(np.clip(obs.get("speed", 0.0) / 4.0, 0, 1))
            if proprioception:
                drive[self.stim_indices["proprioception"]] += np.float32(0.14 * proprioception)
            if taste:
                drive[self.stim_indices["taste"]] += np.float32(0.45 * taste)
            if temperature:
                drive[self.stim_indices["temperature"]] += np.float32(0.82 * temperature)
            if humidity:
                drive[self.stim_indices["humidity"]] += np.float32(0.62 * humidity)
            if wind:
                drive[self.stim_indices["touch"]] += np.float32(0.24 * wind)
            threat = float(np.clip(obs.get("threat", 0.0), 0, 1))
            if threat:
                # Explicit optogenetic-style intervention for the named
                # Moonwalker descending neurons; exposed as such in the UI.
                drive[self.motor_indices["backward"]] += np.float32(0.95 * threat)
            self.arena_sensors = {"odor": odor, "visual": visual, "touch": touch, "taste": taste,
                                  "temperature": temperature, "humidity": humidity, "wind": wind}
            return {
                "external_physics": True,
                "food_distance": float(max(0, obs.get("target_distance", 0.0))),
                "food_bearing": bearing,
                "light_bearing": bearing,
                "touch": touch, "odor": odor, "visual": visual, "taste": taste,
                "temperature": temperature, "humidity": humidity, "wind": wind,
                "reward": float(np.clip(obs.get("reward", 0.0), -2, 4)),
                "gate": int(max(0, obs.get("gate", 0))),
            }
        food_distance, food_bearing, _ = self._target_signal(self.food_x, self.food_y)
        light_distance, light_bearing, _ = self._target_signal(self.light_x, self.light_y)
        odor = float(np.clip(np.exp(-3.2 * food_distance), 0, 1))
        visual = float(np.clip(np.exp(-2.2 * light_distance) * max(0.12, np.cos(light_bearing)), 0, 1))
        wall = min(self.arena_x, 1 - self.arena_x, self.arena_y, 1 - self.arena_y)
        obstacle_center_distance, obstacle_bearing, _ = self._target_signal(self.obstacle_x, self.obstacle_y)
        obstacle_distance = obstacle_center_distance - self.obstacle_r
        touch = float(np.clip((0.12 - min(wall, obstacle_distance)) / 0.12, 0, 1))
        if wall <= obstacle_distance:
            _, avoidance_bearing, _ = self._target_signal(.5, .5)
        else:
            away_x = self.arena_x + (self.arena_x - self.obstacle_x)
            away_y = self.arena_y + (self.arena_y - self.obstacle_y)
            _, avoidance_bearing, _ = self._target_signal(away_x, away_y)
        taste = 1.0 if food_distance < 0.055 else 0.0

        drive[self.stim_indices["olfactory"]] += np.float32(0.04 + 0.48 * odor)
        side_gain = float(np.sin(food_bearing) * odor + 0.55 * np.sin(light_bearing) * visual)
        drive[self.visual_left] += np.float32(0.025 + visual * (0.30 + 0.16 * max(0, side_gain)))
        drive[self.visual_right] += np.float32(0.025 + visual * (0.30 + 0.16 * max(0, -side_gain)))
        if touch > 0:
            drive[self.stim_indices["touch"]] += np.float32(0.65 * touch)
        if self.arena_speed > 0.003:
            drive[self.stim_indices["proprioception"]] += np.float32(min(0.35, self.arena_speed * 18))
        if taste:
            drive[self.stim_indices["taste"]] += np.float32(0.85)
        if self.threat_ticks > 0:
            drive[self.stim_indices["visual"]] += np.float32(0.75)
            self.threat_ticks -= 1
        self.arena_sensors = {"odor": odor, "visual": visual, "touch": touch, "taste": taste}
        return {"food_distance": food_distance, "food_bearing": food_bearing, "light_bearing": light_bearing,
                "obstacle_bearing": obstacle_bearing, "avoidance_bearing": avoidance_bearing,
                "touch": touch, "odor": odor, "visual": visual, "taste": taste}

    def _motor_mean(self, indices: np.ndarray) -> float:
        return float(np.maximum(self.state[indices], 0).mean()) if len(indices) else 0.0

    def _update_arena(self, sensor: dict) -> None:
        if not self.arena_running:
            return
        forward_dn = self._motor_mean(self.motor_indices["forward"])
        backward_dn = self._motor_mean(self.motor_indices["backward"])
        steer_left = self._motor_mean(self.steering_left)
        steer_right = self._motor_mean(self.steering_right)
        dn_turn = np.tanh(2.5 * (steer_right - steer_left))

        if sensor.get("external_physics"):
            obs = self.physics_observation or {}
            bearing = float(sensor["food_bearing"])
            learned_gain = 0.34 + 1.55 * self.odor_memory
            target_turn = float(np.sin(bearing) * learned_gain + .45 * dn_turn)
            if sensor["touch"] > .35:
                target_turn += float(np.sign(-bearing if abs(bearing) > .15 else 1.0) * .75)
            reverse = max(float(obs.get("threat", 0.0)), np.tanh(2.2 * backward_dn))
            base = .38 + .58 * sensor["odor"] * (.35 + self.odor_memory) + .16 * np.tanh(2.0 * forward_dn)
            if reverse > .45:
                base = -.60 * reverse
            target_turn = float(np.clip(target_turn, -0.72, .72))
            self.physics_command = {
                "gain_left": float(np.clip(base - target_turn, -1.2, 1.2)),
                "gain_right": float(np.clip(base + target_turn, -1.2, 1.2)),
                "confidence": float(np.clip(self.odor_memory * sensor["odor"] + .15, 0, 1)),
            }
            self.arena_x = float(obs.get("x", self.arena_x))
            self.arena_y = float(obs.get("y", self.arena_y))
            self.arena_heading = float(obs.get("heading", self.arena_heading))
            self.arena_speed = float(obs.get("speed", self.arena_speed))
            self.arena_turn = target_turn
            self.arena_collisions = int(max(self.arena_collisions, obs.get("collisions", 0)))
            gate = int(sensor.get("gate", self.physics_gate))
            gate_reward = 1.5 * max(0, gate - self.physics_gate)
            reward = float(sensor.get("reward", 0.0)) + gate_reward
            self.physics_gate = max(self.physics_gate, gate)
            self.arena_reward = reward
            self.arena_total_reward += reward
            if gate_reward:
                self.arena_foods += int(gate_reward / 1.5)
            if self.arena_learning and reward > 0:
                self.odor_memory = float(np.clip(
                    self.odor_memory + .055 * reward * sensor["odor"] * (1 - self.odor_memory), 0, 1))
            return

        learned_gain = 0.30 + 2.60 * self.odor_memory
        goal_turn = np.sin(sensor["food_bearing"]) * sensor["odor"] * learned_gain
        light_turn = np.sin(sensor["light_bearing"]) * sensor["visual"] * 0.32
        avoid_turn = 0.0
        if sensor["touch"] > 0:
            avoid_turn = float(np.sin(sensor.get("avoidance_bearing", 0.0)) * (0.9 + 1.4 * sensor["touch"]))
        threat_back = 1.0 if self.threat_ticks > 0 else 0.0
        self.arena_turn = float(np.clip(0.65 * self.arena_turn + 0.35 * (goal_turn + light_turn + .55 * dn_turn + avoid_turn), -1.5, 1.5))
        desired_speed = 0.007 + 0.011 * sensor["odor"] + 0.007 * np.tanh(2 * forward_dn)
        desired_speed -= 0.014 * np.tanh(2 * backward_dn + threat_back)
        self.arena_speed = float(np.clip(0.78 * self.arena_speed + 0.22 * desired_speed, -0.009, 0.022))

        previous_distance = sensor["food_distance"]
        self.arena_heading = self._wrap_angle(self.arena_heading + self.arena_turn * 0.14)
        nx = self.arena_x + np.cos(self.arena_heading) * self.arena_speed
        ny = self.arena_y + np.sin(self.arena_heading) * self.arena_speed
        obstacle_hit = np.hypot(nx - self.obstacle_x, ny - self.obstacle_y) < self.obstacle_r + 0.025
        wall_hit = nx < .025 or nx > .975 or ny < .025 or ny > .975
        collision = bool(obstacle_hit or wall_hit)
        if collision:
            self.arena_heading = self._wrap_angle(self.arena_heading + float(self.rng.choice([-1, 1])) * 1.15)
            self.arena_speed = -0.004
            self.arena_collisions += 1
        else:
            self.arena_x, self.arena_y = float(nx), float(ny)
        new_distance = float(np.hypot(self.food_x - self.arena_x, self.food_y - self.arena_y))
        reward = (previous_distance - new_distance) * 7.0 - (0.28 if collision else 0.0)
        if new_distance < .055:
            reward += 2.0
            self.arena_foods += 1
            self.food_x, self.food_y = self.rng.uniform(.12, .88, 2).tolist()
        self.arena_reward = float(reward)
        self.arena_total_reward += reward
        if self.arena_learning:
            # Reward-gated plasticity in the explicit odor-to-steering decoder.
            positive_prediction_error = max(0.0, reward - self.odor_memory * sensor["odor"] * .02)
            self.odor_memory = float(np.clip(self.odor_memory + .08 * positive_prediction_error * sensor["odor"] * (1 - self.odor_memory), 0, 1))

    def set_stimulus(self, key: str, intensity: float, duration: int) -> None:
        if key in self.stim_indices:
            self.stimulus = key
            self.intensity = float(np.clip(intensity, 0, 3))
            self.remaining = int(np.clip(duration, 1, 200))

    def _game_input(self, drive: np.ndarray) -> None:
        if not self.game_running:
            return
        if self.note_ticks > 2:
            phase = 1.0 - (self.note_ticks - 2) / 12.0
            drive[self.game_sensors[self.note_lane]] += np.float32(.55 + .75 * max(0, phase))
        self.note_ticks -= 1
        if self.note_ticks > 0:
            return

        features = self.state / max(float(np.max(np.abs(self.state))), 1e-4)
        q_values = self.readout @ features
        trials = self.hits + self.misses
        epsilon = max(.08, .50 * np.exp(-trials / 90.0))
        if self.rng.random() < epsilon:
            self.action = int(self.rng.integers(0, 4))
        else:
            self.action = int(np.argmax(q_values + self.rng.normal(0, 1e-5, 4)))
        self.reward = 1.0 if self.action == self.note_lane else -0.25
        if self.action == self.note_lane:
            self.hits += 1
            self.combo += 1
            self.score += 100 + min(self.combo, 50) * 2
        else:
            self.misses += 1
            self.combo = 0
        if self.learning:
            # Dopamine-like reward-gated normalized LMS update on the motor readout.
            # The connectome remains measured/fixed; only this explicit game decoder learns.
            error = self.reward - float(q_values[self.action])
            norm = float(features @ features) + 1e-5
            self.readout[self.action] += np.float32(.42 * error / norm) * features
            self.readout *= np.float32(.9998)
        self.loss_history.append(1.0 if self.action == self.note_lane else 0.0)
        self.note_lane = int(self.rng.integers(0, 4))
        self.note_ticks = 14

    def step(self) -> None:
        drive = np.zeros(self.n, dtype=np.float32)
        if self.remaining > 0 and self.intensity > 0:
            drive[self.stim_indices[self.stimulus]] += np.float32(self.intensity)
            self.remaining -= 1
        arena_sensor = self._arena_input(drive)
        self._game_input(drive)
        propagated = GRAPH @ self.state
        # Stable leaky-rate dynamics on the measured, signed row-normalized graph.
        # The subcritical recurrent gain prevents a sustained real-world stimulus
        # from unrealistically saturating nearly every neuron.
        self.state = np.tanh(np.float32(.70) * self.state + np.float32(.55) * propagated + drive).astype(np.float32)
        self._update_arena(arena_sensor)
        self.tick += 1

    def snapshot(self) -> dict:
        magnitude = np.abs(self.state)
        top_count = min(300, self.n)
        top = np.argpartition(magnitude, -top_count)[-top_count:]
        top = top[np.argsort(magnitude[top])[::-1]]
        region_activity = [float(magnitude[idx].mean()) if len(idx) else 0.0 for idx in self.region_indices]
        stimulus_activity = {
            key: float(magnitude[idx].mean()) if len(idx) else 0.0
            for key, idx in self.stim_indices.items()
        }
        self.activity_history.append(region_activity)
        trials = self.hits + self.misses
        rolling = float(np.mean(self.loss_history)) if self.loss_history else 0.0
        return {
            "type": "state",
            "tick": self.tick,
            "stimulus": self.stimulus,
            "stimulus_remaining": self.remaining,
            "active_neurons": int(np.count_nonzero(magnitude > .05)),
            "mean_activity": float(magnitude.mean()),
            "region_activity": region_activity,
            "stimulus_activity": stimulus_activity,
            "top": {
                "indices": top.tolist(),
                "activity": self.state[top].round(5).tolist(),
                "body_ids": META["body_ids"][top].tolist(),
                "types": META["types"][top].tolist(),
                "superclasses": META["superclasses"][top].tolist(),
            },
            "game": {
                "running": self.game_running,
                "learning": self.learning,
                "note_lane": self.note_lane,
                "note_phase": 1.0 - self.note_ticks / 14.0,
                "action": self.action,
                "reward": self.reward,
                "score": self.score,
                "combo": self.combo,
                "hits": self.hits,
                "misses": self.misses,
                "trials": trials,
                "rolling_accuracy": rolling,
            },
            "arena": {
                "running": self.arena_running,
                "learning": self.arena_learning,
                "physics": "MuJoCo-WASM / NeuroMechFly v2" if self.physics_observation is not None else "fallback 2D model",
                "x": self.arena_x, "y": self.arena_y, "heading": self.arena_heading,
                "speed": self.arena_speed, "turn": self.arena_turn,
                "food": [self.food_x, self.food_y], "light": [self.light_x, self.light_y],
                "obstacle": [self.obstacle_x, self.obstacle_y, self.obstacle_r],
                "threat": self.threat_ticks,
                "sensors": self.arena_sensors,
                "motors": {
                    "DNp09_forward": self._motor_mean(self.motor_indices["forward"]),
                    "MDN_backward": self._motor_mean(self.motor_indices["backward"]),
                    "DNa_left": self._motor_mean(self.steering_left),
                    "DNa_right": self._motor_mean(self.steering_right),
                },
                "reward": self.arena_reward, "total_reward": self.arena_total_reward,
                "food_eaten": self.arena_foods, "collisions": self.arena_collisions,
                "odor_memory": self.odor_memory,
                "gate": self.physics_gate,
                "command": self.physics_command,
            },
        }

    def handle(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "stimulus":
            self.set_stimulus(message.get("key", "visual"), message.get("intensity", 1.0), message.get("duration", 25))
        elif kind == "run":
            self.running = bool(message.get("value", True))
        elif kind == "reset_brain":
            self.reset_brain()
        elif kind == "game":
            self.game_running = bool(message.get("running", self.game_running))
            self.learning = bool(message.get("learning", self.learning))
        elif kind == "reset_learning":
            self.reset_learning()
        elif kind == "arena":
            self.arena_running = bool(message.get("running", self.arena_running))
            self.arena_learning = bool(message.get("learning", self.arena_learning))
        elif kind == "arena_reset":
            self.reset_arena(bool(message.get("keep_learning", True)))
        elif kind == "arena_event":
            event = message.get("event")
            if event == "food":
                self.food_x, self.food_y = self.rng.uniform(.12, .88, 2).tolist()
            elif event == "light":
                self.light_x, self.light_y = self.rng.uniform(.12, .88, 2).tolist()
            elif event == "threat":
                self.threat_ticks = 32
        elif kind == "physics_observation":
            numeric = (
                "x", "y", "z", "heading", "speed", "odor", "visual", "touch", "taste",
                "temperature", "humidity", "wind",
                "target_distance", "target_bearing", "reward", "threat", "collisions", "gate",
            )
            clean = {}
            for key in numeric:
                try:
                    value = float(message.get(key, 0.0))
                    clean[key] = value if np.isfinite(value) else 0.0
                except (TypeError, ValueError):
                    clean[key] = 0.0
            self.physics_observation = clean
            self.arena_running = True
            self.arena_learning = bool(message.get("learning", self.arena_learning))


app = FastAPI(title="Full MaleCNS Brain Lab", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SecurityHeadersMiddleware)


@app.get("/api/health")
def health():
    return {"status": "ok", "neurons": STATS["neurons"], "connections": STATS["connections"]}


@app.get("/api/meta")
def meta():
    return JSONResponse({**STATS, "tick_hz": 8, "limitations": "Connectome topology and synapse counts are measured data; dynamics, stimulus strength and game decoder are explicit models."})


@app.post("/api/analytics/heartbeat")
async def analytics_heartbeat(payload: dict):
    try:
        return ANALYTICS.heartbeat(payload.get("visitor_id"), payload.get("session_id"))
    except ValueError:
        return JSONResponse({"detail": "Invalid anonymous session identifier"}, status_code=400)


@app.get("/api/positions")
def positions():
    return Response(POSITION_BYTES, media_type="application/octet-stream", headers={"X-Neuron-Count": str(STATS["neurons"])})


@app.websocket("/ws/sim")
async def simulation_socket(websocket: WebSocket):
    # Cloudflare supplies this header; fall back to the transport peer for local use.
    client_ip = websocket.headers.get("cf-connecting-ip") or (websocket.client.host if websocket.client else "unknown")
    if not await WS_GATE.acquire(client_ip):
        await websocket.close(code=1013, reason="Connection limit reached")
        return
    await websocket.accept()
    session = BrainSession()
    last_snapshot = None
    dirty = True
    message_times: deque[float] = deque()
    try:
        await websocket.send_json({"type": "meta", **STATS})
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=.125)
                if len(raw.encode("utf-8")) > MAX_WS_MESSAGE_BYTES:
                    await websocket.close(code=1009, reason="Message too large")
                    return
                now = time.monotonic()
                message_times.append(now)
                while message_times and now - message_times[0] > 1.0:
                    message_times.popleft()
                if len(message_times) > MAX_WS_MESSAGES_PER_SECOND:
                    await websocket.close(code=1013, reason="Message rate exceeded")
                    return
                message = json.loads(raw)
                if not isinstance(message, dict):
                    continue
                session.handle(message)
                dirty = True
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass
            if session.running:
                await asyncio.to_thread(session.step)
                dirty = True
            if dirty or last_snapshot is None:
                last_snapshot = session.snapshot()
                dirty = False
            await websocket.send_json(last_snapshot)
    except WebSocketDisconnect:
        return
    finally:
        await WS_GATE.release(client_ip)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
