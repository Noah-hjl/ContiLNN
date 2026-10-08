#!/usr/bin/env python3
# Editor: Jialei.He
"""CT, MRI, and PET readers must remain explicit and mutually isolated."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from contilnn import cli
from contilnn.data import ct, mri, pet
from contilnn.data.names import parse_ct_stem, parse_mri_stem, parse_pet_stem


VALID = {
    "CT": "L506_105_0",
    "MRI": "IXI385-HH-2078-T2_58",
    "PET": "Patient_41_105_0",
}
PARSERS = {"CT": parse_ct_stem, "MRI": parse_mri_stem, "PET": parse_pet_stem}


class ModalityGrammarTest(unittest.TestCase):
    def test_valid_examples_have_expected_case_and_slice_fields(self):
        parsed_ct = parse_ct_stem(VALID["CT"])
        self.assertEqual(
            (parsed_ct.case_id, parsed_ct.slice_index, parsed_ct.auxiliary_index),
            ("L506", 105, 0),
        )
        parsed_mri = parse_mri_stem(VALID["MRI"])
        self.assertEqual(
            (parsed_mri.case_id, parsed_mri.slice_index, parsed_mri.auxiliary_index),
            ("IXI385-HH-2078-T2", 58, None),
        )
        parsed_pet = parse_pet_stem(VALID["PET"])
        self.assertEqual(
            (parsed_pet.patient, parsed_pet.z, parsed_pet.patch),
            ("Patient_41", 105, 0),
        )

    def test_pet_case_id_does_not_collapse_to_the_ct_grammar(self):
        parsed = parse_pet_stem("Patient_101_0")
        self.assertEqual(parsed.patient, "Patient_101")
        self.assertEqual(parsed.z, 0)
        self.assertIsNone(parsed.patch)

    def test_cross_modality_names_are_rejected(self):
        for modality, parser in PARSERS.items():
            for other, value in VALID.items():
                if modality == other:
                    continue
                with self.subTest(parser=modality, input=other):
                    with self.assertRaises(ValueError):
                        parser(value)


class ReaderIsolationTest(unittest.TestCase):
    def test_cli_selects_the_matching_reader_pair_only(self):
        expected = {
            "CT": (ct.CtSequenceTrainDataset, ct.CtCaseDataset),
            "MRI": (mri.MriSequenceTrainDataset, mri.MriCaseDataset),
            "PET": (pet.PetSequenceTrainDataset, pet.PetCaseDataset),
        }
        for modality, classes in expected.items():
            with self.subTest(modality=modality):
                self.assertEqual(cli._reader_classes(modality), classes)

    def test_reader_counts_match_the_reported_cohorts(self):
        self.assertEqual(ct.EXPECTED_TRAIN_CASES, 8)
        self.assertEqual(ct.EXPECTED_CASES, {"validation": 1, "test": 1})
        self.assertEqual(mri.EXPECTED_TRAIN_CASES, 405)
        self.assertEqual(mri.EXPECTED_CASES, {"validation": 58, "test": 114})
        self.assertEqual(pet.EXPECTED_TRAIN_PATIENTS, 120)
        self.assertEqual(pet.EXPECTED_CASES, {"validation": 10, "test": 29})

    def test_public_constructors_do_not_expose_count_bypasses(self):
        classes = (
            ct.CtSequenceTrainDataset,
            ct.CtCaseDataset,
            mri.MriSequenceTrainDataset,
            mri.MriCaseDataset,
            pet.PetSequenceTrainDataset,
            pet.PetCaseDataset,
        )
        for reader in classes:
            with self.subTest(reader=reader.__name__):
                self.assertNotIn(
                    "enforce_expected_count", inspect.signature(reader).parameters
                )

    def test_pet_evaluation_reader_preserves_29_distinct_patients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lq = root / "PET" / "test" / "LQ"
            hq = root / "PET" / "test" / "HQ"
            lq.mkdir(parents=True)
            hq.mkdir(parents=True)
            for patient in range(1, 30):
                name = f"Patient_{patient}_0.nii"
                (lq / name).touch()
                (hq / name).touch()

            dataset = pet.PetCaseDataset(str(root), "test")
            self.assertEqual(len(dataset), 29)
            self.assertEqual(dataset.case_keys[0], "PET:Patient_1")
            self.assertEqual(len(set(dataset.case_keys)), 29)


if __name__ == "__main__":
    unittest.main()
