#!/usr/bin/env python3
# Editor: Jialei.He
"""The bundled protocols are the complete public experiment schema."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from contilnn.config import ProtocolError, load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = (
    Path("configs/01_rwkv_main/protocol.json"),
    Path("configs/02_dasmamba_transfer/protocol.json"),
)
MODALITY_COUNTS = {
    "CT": {"train_cases": 8, "validation_cases": 1, "test_cases": 1},
    "MRI": {"train_cases": 405, "validation_cases": 58, "test_cases": 114},
    "PET": {"train_cases": 120, "validation_cases": 10, "test_cases": 29},
}


class ProtocolTest(unittest.TestCase):
    def test_only_the_two_reported_protocols_are_bundled(self):
        files = tuple(
            sorted(path.relative_to(ROOT) for path in (ROOT / "configs").rglob("*.json"))
        )
        self.assertEqual(files, PROTOCOL_FILES)

    def test_protocol_aliases_resolve_to_the_reported_experiments(self):
        rwkv = load_protocol("rwkv")
        dasmamba = load_protocol("dasmamba")
        self.assertEqual(rwkv.experiment, "contilnn_rwkv_main")
        self.assertEqual(rwkv.backbone, "Restore-RWKV")
        self.assertEqual(dasmamba.experiment, "contilnn_dasmamba_transfer")
        self.assertEqual(dasmamba.backbone, "DASMamba")

    def test_shared_scope_and_cohort_counts_are_exact(self):
        protocols = [load_protocol(ROOT / path) for path in PROTOCOL_FILES]
        self.assertEqual([item.payload["paper_order"] for item in protocols], [1, 2])
        for protocol in protocols:
            payload = protocol.as_dict()
            self.assertEqual(payload["editor"], "Jialei.He")
            self.assertEqual(payload["operator"], "bidirectional_cfc")
            self.assertEqual(payload["num_seeds"], 5)
            self.assertEqual(tuple(payload["modalities"]), ("CT", "MRI", "PET"))
            self.assertEqual(payload["data"]["sequence_length"], 7)
            self.assertEqual(payload["data"]["patch_size"], 128)
            self.assertEqual(payload["data"]["slice_index_gap"], 1)
            self.assertIs(payload["data"]["retain_short_tail"], True)
            self.assertEqual(payload["data"]["evaluation_offsets"], [0, 3])
            self.assertEqual(payload["data"]["overlap_fusion"], "mean")
            for modality, counts in MODALITY_COUNTS.items():
                for name, expected in counts.items():
                    self.assertEqual(
                        payload["modalities"][modality][name], expected
                    )

    def test_rwkv_objective_matches_the_manuscript(self):
        objective = load_protocol("rwkv").payload["objective"]
        self.assertEqual(
            objective,
            {
                "reconstruction": 1.0,
                "consistency": 0.30,
                "second_order_multiplier": 2.0,
                "teacher_distillation": 0.05,
            },
        )

    def test_unreported_fields_are_rejected(self):
        payload = copy.deepcopy(load_protocol("rwkv").as_dict())
        payload["extra_control"] = {"operator": "other"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "unexpected"):
                load_protocol(path)

    def test_changed_primary_data_definition_is_rejected(self):
        payload = copy.deepcopy(load_protocol("rwkv").as_dict())
        payload["data"]["slice_index_gap"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "slice_index_gap"):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
