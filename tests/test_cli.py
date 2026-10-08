#!/usr/bin/env python3
# Editor: Jialei.He
"""CPU-only tests for the public command-line contract."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from contilnn import cli
from contilnn.config import ProtocolError, load_protocol


class ParserTest(unittest.TestCase):
    def test_only_supported_commands_are_registered(self):
        parser = cli.build_parser()
        choices = parser._subparsers._group_actions[0].choices
        self.assertEqual(
            set(choices), {"inspect-protocol", "train", "evaluate"}
        )

    def test_data_commands_require_explicit_modality(self):
        parser = cli.build_parser()
        commands = (
            [
                "train",
                "--protocol",
                "rwkv",
                "--data-root",
                "/data",
                "--stage-a-checkpoint",
                "/stage-a.pth",
                "--output-dir",
                "/output",
                "--seed",
                "7",
            ],
            [
                "evaluate",
                "--protocol",
                "rwkv",
                "--data-root",
                "/data",
                "--checkpoint",
                "/model.pth",
                "--output-dir",
                "/output",
            ],
        )
        for command in commands:
            with self.subTest(command=command[0]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(command)

    def test_unknown_modality_is_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "Unsupported modality"):
            cli._reader_classes("XRAY")

    def test_rwkv_protocol_values_cannot_be_overridden_from_cli(self):
        args = cli.build_parser().parse_args(
            [
                "train",
                "--protocol",
                "rwkv",
                "--data-root",
                "/data",
                "--modality",
                "PET",
                "--stage-a-checkpoint",
                "/stage-a.pth",
                "--output-dir",
                "/output",
                "--seed",
                "7",
                "--learning-rate-base",
                "0.1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            cli._training_settings(load_protocol("rwkv"), args)

    def test_rwkv_training_settings_resolve_all_fixed_values(self):
        args = cli.build_parser().parse_args(
            [
                "train",
                "--protocol",
                "rwkv",
                "--data-root",
                "/data",
                "--modality",
                "PET",
                "--stage-a-checkpoint",
                "/stage-a.pth",
                "--output-dir",
                "/output",
                "--seed",
                "7",
            ]
        )
        settings = cli._training_settings(load_protocol("rwkv"), args)
        self.assertEqual(settings.experiment, "contilnn_rwkv_main")
        self.assertEqual(settings.modality, "PET")
        self.assertEqual(settings.max_epochs, 60)
        self.assertAlmostEqual(settings.gradient_clip_norm, 1.0)
        self.assertAlmostEqual(settings.consistency_weight, 0.30)
        self.assertAlmostEqual(settings.distillation_weight, 0.05)

    def test_dasmamba_unfixed_learning_rates_must_be_explicit(self):
        args = cli.build_parser().parse_args(
            [
                "train",
                "--protocol",
                "dasmamba",
                "--data-root",
                "/data",
                "--modality",
                "MRI",
                "--stage-a-checkpoint",
                "/stage-a.pth",
                "--output-dir",
                "/output",
                "--seed",
                "7",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--learning-rate-base"):
            cli._training_settings(load_protocol("dasmamba"), args)


class ProtocolInspectionTest(unittest.TestCase):
    def test_inspection_prints_the_validated_protocol(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = cli.main(["inspect-protocol", "--protocol", "rwkv"])
        self.assertEqual(status, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["experiment"], "contilnn_rwkv_main")
        self.assertEqual(payload["editor"], "Jialei.He")
        self.assertEqual(payload["data"]["sequence_length"], 7)


class CheckpointV2Test(unittest.TestCase):
    def test_evaluation_checkpoint_matches_selected_protocol_and_modality(self):
        protocol = load_protocol("rwkv")
        payload = {
            "format": "contilnn-training-v2",
            "experiment": protocol.experiment,
            "backbone": protocol.backbone,
            "modality": "PET",
            "model": {"weight": torch.ones(1)},
        }
        model = mock.Mock()
        with mock.patch("torch.load", return_value=payload):
            cli._load_checkpoint(model, Path("best.pth"), protocol, "PET")
            model.load_state_dict.assert_called_once_with(payload["model"], strict=True)
            model.reset_mock()
            with self.assertRaisesRegex(ValueError, "Checkpoint modality"):
                cli._load_checkpoint(model, Path("best.pth"), protocol, "MRI")
            model.load_state_dict.assert_not_called()
        for field, wrong_value in (
            ("experiment", "contilnn_dasmamba_transfer"),
            ("backbone", "DASMamba"),
            ("modality", None),
        ):
            with self.subTest(field=field), mock.patch(
                "torch.load", return_value={**payload, field: wrong_value}
            ):
                with self.assertRaisesRegex(ValueError, f"Checkpoint {field}"):
                    cli._load_checkpoint(model, Path("best.pth"), protocol, "PET")
                model.load_state_dict.assert_not_called()

    def test_stage_b_checkpoint_requires_v2_model_envelope(self):
        state = {"base.weight": torch.ones(1), "lnn.weight": torch.zeros(1)}
        self.assertIs(
            cli._checkpoint_state(
                {"format": "contilnn-training-v2", "model": state}
            ),
            state,
        )

    def test_legacy_and_generic_checkpoint_shapes_are_rejected(self):
        invalid = (
            {"base.weight": object()},
            {"format": "contilnn-training-v1", "model": {"base.weight": object()}},
            {"state_dict": {"base.weight": object()}},
            {"model_state_dict": {"base.weight": object()}},
            {
                "format": "contilnn-training-v2",
                "model": {"module.base.weight": object()},
            },
        )
        for payload in invalid:
            with self.subTest(keys=tuple(payload)):
                with self.assertRaises((RuntimeError, TypeError, ValueError)):
                    cli._checkpoint_state(payload)


class _FakeModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeDevice:
    type = "cuda"


class _FakeCaseDataset:
    calls = []

    def __init__(self, data_root, split):
        type(self).calls.append((data_root, split))

    def __len__(self):
        return 1


class EvaluationCommandTest(unittest.TestCase):
    def setUp(self):
        _FakeCaseDataset.calls = []

    def test_evaluation_uses_one_selected_reader_and_writes_standard_outputs(self):
        protocol = load_protocol("rwkv")
        summary = {"cases": 1, "psnr": 30.0, "ssim": 0.9, "rmse": 0.1}
        per_case = [
            {"case": "CT:L506", "slices": 1, "psnr": 30.0, "ssim": 0.9, "rmse": 0.1}
        ]
        per_slice = [
            {
                "case": "CT:L506",
                "slice": "L506_0",
                "psnr": 30.0,
                "ssim": 0.9,
                "rmse": 0.1,
            }
        ]
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"test-placeholder")
            with mock.patch.object(
                cli, "_reader_classes", return_value=(object, _FakeCaseDataset)
            ) as reader_selector, mock.patch.object(
                cli, "_build_model", return_value=model
            ), mock.patch.object(
                cli, "_load_checkpoint"
            ) as load_checkpoint, mock.patch.object(
                cli, "_resolve_device", return_value=_FakeDevice()
            ), mock.patch(
                "contilnn.evaluation.evaluate_case_dataset",
                return_value=(summary, per_case, per_slice),
            ) as evaluator:
                record = cli._evaluate(
                    protocol=protocol,
                    data_root=root / "data",
                    modality="CT",
                    split="test",
                    checkpoint=checkpoint,
                    output_dir=output,
                    device_specification="cuda",
                    backbone_factory=None,
                )

            reader_selector.assert_called_once_with("CT")
            self.assertEqual(_FakeCaseDataset.calls, [(str(root / "data"), "test")])
            load_checkpoint.assert_called_once_with(model, checkpoint, protocol, "CT")
            evaluator.assert_called_once()
            self.assertEqual(evaluator.call_args.kwargs["sequence_length"], 7)
            self.assertEqual(record["summary"], summary)
            self.assertTrue((output / "per_case.csv").is_file())
            self.assertTrue((output / "per_slice.csv").is_file())
            stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["checkpoint"], "model.pth")
            self.assertEqual(stored["protocol"], "contilnn_rwkv_main")
            self.assertNotIn(str(root), json.dumps(stored))

    def test_main_dispatches_training_without_importing_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "stage-a.pth"
            checkpoint.write_bytes(b"placeholder")
            stdout = io.StringIO()
            with mock.patch.object(
                cli,
                "_train",
                return_value={"status": "PASS", "best_validation_psnr": 31.0},
            ) as train, redirect_stdout(stdout):
                status = cli.main(
                    [
                        "train",
                        "--protocol",
                        "rwkv",
                        "--data-root",
                        str(root / "data"),
                        "--modality",
                        "PET",
                        "--stage-a-checkpoint",
                        str(checkpoint),
                        "--output-dir",
                        str(root / "output"),
                        "--seed",
                        "7",
                    ]
                )
            self.assertEqual(status, 0)
            train.assert_called_once()
            self.assertEqual(json.loads(stdout.getvalue())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
