from __future__ import annotations

import numpy as np
from scipy import sparse


def simulate_lif(
    adjacency: sparse.csr_matrix,
    steps: int = 240,
    stimulus: float = 1.4,
    stimulus_until: int = 35,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a deliberately simple LIF-like model on a connectome-derived graph.

    The wiring is biological; time constants, thresholds and input mapping are a model.
    """
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    voltage = rng.normal(0, 0.02, n).astype(np.float32)
    spikes = np.zeros((steps, n), dtype=np.uint8)
    mean_voltage = np.empty(steps, dtype=np.float32)
    input_count = max(4, n // 100)

    for t in range(steps):
        recurrent = adjacency @ spikes[t - 1].astype(np.float32) if t else 0.0
        current = 1.15 * recurrent + rng.normal(0, 0.015, n)
        if t < stimulus_until:
            current[:input_count] += stimulus
        voltage = 0.91 * voltage + current
        fired = voltage > 1.0
        spikes[t, fired] = 1
        voltage[fired] = 0
        mean_voltage[t] = float(voltage.mean())

    return spikes, mean_voltage


class ConnectomeReservoir:
    """Small connectome-topology reservoir with a trained linear motor readout."""

    def __init__(self, adjacency: sparse.csr_matrix, seed: int = 4):
        self.a = adjacency.astype(np.float32)
        self.n = adjacency.shape[0]
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(self.n, dtype=np.float32)
        self.sensor_ids = np.array_split(np.arange(min(80, self.n)), 4)
        self.readout = np.zeros((2, self.n + 1), dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(0)

    def step(self, observation: np.ndarray) -> np.ndarray:
        drive = np.zeros(self.n, dtype=np.float32)
        for value, ids in zip(observation[:4], self.sensor_ids):
            drive[ids] = float(value)
        self.state = np.tanh(0.82 * self.state + 1.35 * (self.a @ self.state) + drive)
        features = np.append(self.state, 1.0)
        return self.readout @ features

    def train_motor_readout(self, samples: int = 900, ridge: float = 1e-2) -> float:
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        self.reset()
        lane_xy = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]], dtype=np.float32)
        for _ in range(samples):
            lane = int(self.rng.integers(0, 4))
            obs = np.zeros(4, dtype=np.float32)
            obs[lane] = 1
            self.step(obs)
            features.append(np.append(self.state, 1.0))
            targets.append(lane_xy[lane])
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float32)
        self.readout = np.linalg.solve(gram, x.T @ y).T.astype(np.float32)
        pred = x @ self.readout.T
        return float(np.mean(np.linalg.norm(pred - y, axis=1)))

