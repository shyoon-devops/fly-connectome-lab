from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "webapp" / "static" / "nmf" / "game" / "assets"


def validate_assets() -> dict:
    meta = json.loads((ASSETS / "model_meta.json").read_text(encoding="utf-8"))
    xml_path = ASSETS / "model" / "fly.xml"
    root = ET.parse(xml_path).getroot()
    option = root.find("option")
    assert option is not None
    assert math.isclose(float(option.attrib["timestep"]), 1e-4)
    gravity = [float(x) for x in option.attrib["gravity"].split()]
    assert gravity == [0.0, 0.0, -9810.0]
    assert meta["nq"] == 73 and meta["nu"] == 48
    assert len(meta["actuators"]) == 42
    assert len(meta["adhesion"]) == 6
    assert len(meta["ctrl_index_by_leg_dof"]) == 6
    assert all(len(row) == 7 for row in meta["ctrl_index_by_leg_dof"])
    meshes = {e.attrib["file"] for e in root.findall(".//mesh") if "file" in e.attrib}
    missing = [name for name in meshes if not (ASSETS / "model" / name).is_file()]
    assert not missing, missing
    contact_pairs = root.findall(".//contact/pair")
    assert len(contact_pairs) > 100
    return {
        "nq": meta["nq"], "nu": meta["nu"], "timestep": meta["timestep"],
        "gravity_mm_s2": gravity[2], "meshes": len(meshes), "contact_pairs": len(contact_pairs),
    }


def validate_brain_bridge() -> dict:
    from webapp.server import BrainSession, GRAPH, META, STATS

    assert GRAPH.shape == (166_700, 166_700)
    assert GRAPH.nnz == 25_582_938
    assert STATS["synapses"] == 124_177_632
    types = META["types"]
    expected = {"DNp09": 2, "MDN": 4, "DNa01": 2, "DNa02": 2}
    actual = {name: int(np.count_nonzero(types == name)) for name in expected}
    assert actual == expected, actual

    session = BrainSession(seed=17)
    session.handle({"type": "run", "value": True})
    observation = {
        "type": "physics_observation", "x": 1.0, "y": 0.0, "z": 1.0,
        "heading": 0.0, "speed": 1.2, "odor": 0.62, "visual": 0.9,
        "touch": 0.0, "taste": 0.0, "target_distance": 9.0,
        "target_bearing": 0.4, "reward": 0.03, "gate": 0,
        "collisions": 0, "learning": True,
    }
    session.handle(observation)
    for _ in range(80):
        session.step()
    snap = session.snapshot()
    active = snap["active_neurons"]
    assert 5_000 < active < 90_000, active
    command = snap["arena"]["command"]
    assert all(math.isfinite(v) for v in command.values())
    assert -1.2 <= command["gain_left"] <= 1.2
    assert -1.2 <= command["gain_right"] <= 1.2
    before = session.odor_memory
    session.handle({**observation, "reward": 1.0, "gate": 1, "taste": 1.0})
    session.step()
    assert session.odor_memory > before
    return {
        "neurons": GRAPH.shape[0], "connections": GRAPH.nnz, "synapses": STATS["synapses"],
        "motor_cells": actual, "active_at_steady_stimulus": active,
        "command": command, "memory_before_reward": before, "memory_after_reward": session.odor_memory,
    }


def main() -> None:
    notice = ROOT / "webapp" / "static" / "nmf" / "NOTICE.txt"
    assert notice.is_file() and "NeuroMechFly v2" in notice.read_text(encoding="utf-8")
    result = {"physics": validate_assets(), "brain_bridge": validate_brain_bridge()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
