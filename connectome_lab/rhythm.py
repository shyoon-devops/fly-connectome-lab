from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pygame

from connectome_lab.data import load_graph
from connectome_lab.simulation import ConnectomeReservoir


WIDTH, HEIGHT = 1100, 760
CENTERS = [(310, 250), (790, 250), (310, 545), (790, 545)]
COLORS = [(105, 244, 187), (111, 174, 255), (255, 118, 143), (236, 188, 88)]


@dataclass
class Beat:
    lane: int
    due: float
    judged: bool = False


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connectome Saber — MaleCNS topology demo")
    font = pygame.font.SysFont("segoeui", 25)
    small = pygame.font.SysFont("segoeui", 17)
    title = pygame.font.SysFont("segoeui", 38, bold=True)
    clock = pygame.time.Clock()

    graph = load_graph()
    n = min(360, graph.neuron_count)
    model = ConnectomeReservoir(graph.adjacency(n))
    error = model.train_motor_readout(samples=800)

    rng = np.random.default_rng(22)
    start = time.perf_counter()
    beats = [Beat(int(rng.integers(0, 4)), start + 2.2 + i * .72) for i in range(400)]
    cursor = np.array([WIDTH / 2, HEIGHT / 2], dtype=np.float32)
    score = combo = misses = 0
    auto = True
    activity = []
    frame_count = 0
    test_frames = int(os.environ.get("CONNECTOME_SABER_TEST_FRAMES", "0"))
    running = True

    while running:
        now = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_a: auto = True
                if event.key == pygame.K_m: auto = False
                if event.key == pygame.K_r:
                    start = now; score = combo = misses = 0
                    beats = [Beat(int(rng.integers(0, 4)), start + 2.2 + i * .72) for i in range(400)]

        upcoming = next((b for b in beats if not b.judged), None)
        obs = np.zeros(4, dtype=np.float32)
        if upcoming and 0 < upcoming.due - now < 1.25:
            obs[upcoming.lane] = max(0, 1.25 - (upcoming.due - now)) / 1.25
        output = model.step(obs)
        activity.append(float(np.mean(np.abs(model.state))))
        activity = activity[-240:]

        if auto:
            tx = WIDTH / 2 + float(np.clip(output[0], -1.2, 1.2)) * 240
            ty = HEIGHT / 2 - float(np.clip(output[1], -1.2, 1.2)) * 150
            cursor += .18 * (np.array([tx, ty]) - cursor)
        else:
            keys = pygame.key.get_pressed()
            cursor += np.array([keys[pygame.K_RIGHT] - keys[pygame.K_LEFT], keys[pygame.K_DOWN] - keys[pygame.K_UP]]) * 9

        for beat in beats:
            if not beat.judged and now >= beat.due:
                beat.judged = True
                dist = np.linalg.norm(cursor - np.asarray(CENTERS[beat.lane]))
                if dist < 135:
                    combo += 1; score += 100 + min(combo, 50) * 2
                else:
                    combo = 0; misses += 1
                break

        screen.fill((6, 11, 10))
        pygame.draw.circle(screen, (18, 38, 32), (550, 380), 510)
        screen.blit(title.render("CONNECTOME SABER", True, (224, 247, 239)), (35, 25))
        screen.blit(small.render("real MaleCNS wiring · modeled dynamics, sensors and motor readout", True, (128, 158, 148)), (38, 72))
        screen.blit(font.render(f"SCORE {score:07d}    COMBO {combo:03d}    MISS {misses}", True, (220, 238, 232)), (660, 35))
        mode = "AUTO / connectome reservoir" if auto else "MANUAL / arrow keys"
        screen.blit(small.render(f"[A] auto  [M] manual  [R] reset  [ESC] quit   •   {mode}", True, (140, 180, 166)), (35, 720))

        for lane, center in enumerate(CENTERS):
            pygame.draw.circle(screen, (25, 43, 38), center, 105, 4)
            pygame.draw.circle(screen, COLORS[lane], center, 6)

        if upcoming:
            remaining = upcoming.due - now
            radius = int(np.clip(25 + remaining * 100, 25, 160))
            color = COLORS[upcoming.lane]
            pygame.draw.circle(screen, color, CENTERS[upcoming.lane], radius, 7)
            if remaining < 0.32:
                alpha = int(155 + 100 * math.sin(now * 30))
                glow = tuple(min(255, c + alpha // 5) for c in color)
                pygame.draw.circle(screen, glow, CENTERS[upcoming.lane], max(18, radius - 15), 3)

        pygame.draw.circle(screen, (245, 250, 248), cursor.astype(int), 18, 3)
        pygame.draw.line(screen, (245, 250, 248), cursor.astype(int) - (28, 28), cursor.astype(int) + (28, 28), 3)

        if len(activity) > 2:
            pts = [(35 + i * 2, 690 - int(v * 400)) for i, v in enumerate(activity)]
            pygame.draw.lines(screen, (74, 229, 167), False, pts, 2)
            screen.blit(small.render("mean neural activity", True, (104, 148, 133)), (35, 650))
        screen.blit(small.render(f"motor readout fit error: {error:.4f} · {n} real-connectome nodes", True, (104, 148, 133)), (750, 685))

        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
        if test_frames and frame_count >= test_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
