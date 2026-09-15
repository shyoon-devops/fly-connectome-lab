from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "interactive_graph.npz"


@dataclass
class ConnectomeGraph:
    body_ids: np.ndarray
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    labels: np.ndarray

    @property
    def neuron_count(self) -> int:
        return int(self.body_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.weight.size)

    def adjacency(self, limit: int | None = None) -> sparse.csr_matrix:
        n = self.neuron_count if limit is None else min(limit, self.neuron_count)
        keep = (self.source < n) & (self.target < n)
        values = np.log1p(self.weight[keep]).astype(np.float32)
        matrix = sparse.csr_matrix(
            (values, (self.target[keep], self.source[keep])), shape=(n, n)
        )
        scale = np.asarray(matrix.sum(axis=1)).ravel()
        scale[scale == 0] = 1
        return sparse.diags(1.0 / scale) @ matrix


def load_graph(path: Path = PROCESSED) -> ConnectomeGraph:
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared graph not found at {path}. Run .\\setup.ps1 first."
        )
    data = np.load(path, allow_pickle=False)
    return ConnectomeGraph(
        body_ids=data["body_ids"],
        source=data["source"],
        target=data["target"],
        weight=data["weight"],
        labels=data["labels"],
    )

