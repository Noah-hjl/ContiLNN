#!/usr/bin/env python3
# Editor: Jialei.He
"""Unit tests for the published loss and image-metric definitions."""

from __future__ import annotations

import math
import unittest

import torch

from contilnn.evaluation.metrics import (
    compute_MSE,
    compute_PSNR,
    compute_RMSE,
    compute_SSIM,
    compute_measure,
)
from contilnn.training.losses import (
    consistency_terms,
    contilnn_objective,
    reconstruction_terms,
)


class ConsistencyLossTest(unittest.TestCase):
    def test_first_and_second_order_terms_match_the_definition(self):
        prediction = torch.tensor([0.0, 1.0, 4.0, 9.0]).view(1, 4, 1, 1, 1)
        target = torch.zeros_like(prediction)
        terms = consistency_terms(prediction, target, second_order_multiplier=2.0)
        expected_first = torch.tensor((1.0 + 3.0 + 5.0) / 3.0)
        expected_second = torch.tensor(2.0)
        self.assertTrue(torch.allclose(terms["first_order"], expected_first))
        self.assertTrue(torch.allclose(terms["second_order"], expected_second))
        self.assertTrue(
            torch.allclose(
                terms["consistency"], expected_first + 2.0 * expected_second
            )
        )

    def test_short_sequences_have_well_defined_zero_terms(self):
        one = torch.ones(1, 1, 1, 2, 2)
        two = torch.ones(1, 2, 1, 2, 2)
        one_terms = consistency_terms(one, one)
        two_terms = consistency_terms(two, two)
        self.assertEqual(one_terms["first_order"].item(), 0.0)
        self.assertEqual(one_terms["second_order"].item(), 0.0)
        self.assertEqual(two_terms["second_order"].item(), 0.0)

    def test_consistency_rejects_non_sequence_or_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            consistency_terms(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
        with self.assertRaises(ValueError):
            consistency_terms(
                torch.zeros(1, 2, 1, 2, 2),
                torch.zeros(1, 3, 1, 2, 2),
            )


class ReconstructionLossTest(unittest.TestCase):
    def test_rwkv_reconstruction_is_spatial_l1(self):
        prediction = torch.zeros(1, 2, 1, 2, 2)
        target = torch.ones_like(prediction)
        terms = reconstruction_terms(prediction, target, backbone="Restore-RWKV")
        self.assertEqual(terms["spatial_reconstruction"].item(), 1.0)
        self.assertEqual(terms["fourier_reconstruction"].item(), 0.0)
        self.assertEqual(terms["reconstruction"].item(), 1.0)

    def test_dasmamba_retains_spatial_and_fourier_terms(self):
        prediction = torch.zeros(1, 1, 1, 2, 2)
        target = torch.ones_like(prediction)
        terms = reconstruction_terms(
            prediction, target, backbone="DASMamba", fourier_weight=0.5
        )
        expected_spectral = torch.mean(
            torch.abs(
                torch.fft.fft2(prediction.float(), dim=(-2, -1))
                - torch.fft.fft2(target.float(), dim=(-2, -1))
            )
        )
        self.assertTrue(
            torch.allclose(
                terms["reconstruction"],
                torch.tensor(1.0) + 0.5 * expected_spectral,
            )
        )

    def test_complete_objective_uses_reported_weights(self):
        prediction = torch.zeros(1, 3, 1, 2, 2)
        target = torch.ones_like(prediction)
        teacher = torch.full_like(prediction, 0.5)
        terms = contilnn_objective(
            prediction,
            target,
            teacher,
            backbone="Restore-RWKV",
            consistency_weight=0.30,
            second_order_multiplier=2.0,
            distillation_weight=0.05,
        )
        expected = (
            terms["reconstruction"]
            + 0.30 * terms["consistency"]
            + 0.05 * terms["distillation"]
        )
        self.assertTrue(torch.allclose(terms["total"], expected))
        self.assertEqual(terms["distillation"].item(), 0.5)

    def test_enabled_distillation_requires_teacher_prediction(self):
        value = torch.zeros(1, 3, 1, 2, 2)
        with self.assertRaises(TypeError):
            contilnn_objective(
                value,
                value,
                None,
                backbone="Restore-RWKV",
                distillation_weight=0.05,
            )


class ImageMetricTest(unittest.TestCase):
    def test_mse_rmse_psnr_and_measure_agree(self):
        target = torch.zeros(1, 1, 4, 4)
        prediction = torch.full_like(target, 0.5)
        self.assertAlmostEqual(compute_MSE(target, prediction).item(), 0.25)
        self.assertAlmostEqual(compute_RMSE(target, prediction), 0.5)
        self.assertAlmostEqual(
            compute_PSNR(target, prediction, data_range=1.0), 6.020600, places=5
        )
        psnr, ssim, rmse = compute_measure(target, prediction, data_range=1.0)
        self.assertAlmostEqual(psnr, 6.020600, places=5)
        self.assertTrue(math.isfinite(ssim))
        self.assertAlmostEqual(rmse, 0.5)

    def test_rectangular_2d_ssim_is_one_for_identical_images(self):
        image = torch.linspace(0.0, 1.0, steps=7 * 11).reshape(7, 11)
        self.assertAlmostEqual(
            compute_SSIM(image, image.clone(), data_range=1.0), 1.0, places=6
        )

    def test_ssim_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            compute_SSIM(
                torch.zeros(7, 11), torch.zeros(7, 10), data_range=1.0
            )


if __name__ == "__main__":
    unittest.main()
