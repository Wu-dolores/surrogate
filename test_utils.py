"""
Unit tests for utility functions.

Run with: pytest test_utils.py
"""

import numpy as np
import pytest
from module.utils import (
    zapply,
    zfit,
    cumtrapz_batch_np,
    cwp_rw_from_q_logp_np,
    alpha_full_column,
    alpha_bottom_window
)


class TestNormalization:
    """Tests for normalization functions."""

    def test_zapply_basic(self):
        """Test basic z-score normalization."""
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        mu = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
        std = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

        result = zapply(x, mu, std)
        expected = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_zfit_consistency(self):
        """Test that zfit and zapply are consistent."""
        x = np.random.randn(100, 5).astype(np.float32)
        mu, std = zfit(x)
        x_norm = zapply(x, mu, std)

        # Normalized data should have ~0 mean and ~1 std
        assert np.abs(x_norm.mean()) < 0.1
        assert np.abs(x_norm.std() - 1.0) < 0.1


class TestIntegration:
    """Tests for numerical integration functions."""

    def test_cumtrapz_linear(self):
        """Test cumulative trapezoidal integration on linear function."""
        # Integrate y = x from 0 to 1
        x = np.linspace(0, 1, 11, dtype=np.float32)[None, :]
        y = x.copy()

        result = cumtrapz_batch_np(y, x)

        # Analytical result: integral of x is x^2/2
        expected = x ** 2 / 2

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_cumtrapz_constant(self):
        """Test integration of constant function."""
        x = np.linspace(0, 1, 11, dtype=np.float32)[None, :]
        y = np.ones_like(x) * 2.0

        result = cumtrapz_batch_np(y, x)

        # Integral of constant 2 is 2*x
        expected = 2.0 * x

        np.testing.assert_allclose(result, expected, rtol=1e-4)


class TestWaterPath:
    """Tests for water path calculations."""

    def test_cwp_rw_sum(self):
        """Test that CWP + RW equals total column water."""
        B, N = 10, 20
        q = np.random.rand(B, N).astype(np.float32) * 0.01
        logp = np.linspace(2, 5, N, dtype=np.float32)[None, :].repeat(B, axis=0)

        cwp, rw = cwp_rw_from_q_logp_np(q, logp)

        # CWP at surface should equal total
        total = cwp[:, -1]

        # CWP + RW should equal total at all levels
        for i in range(N):
            np.testing.assert_allclose(
                cwp[:, i] + rw[:, i],
                total,
                rtol=1e-4
            )


class TestAlphaWeighting:
    """Tests for alpha weighting functions."""

    def test_alpha_full_column_range(self):
        """Test that alpha is in [0, 1] range."""
        B, N = 5, 30
        logp = np.linspace(2, 5, N, dtype=np.float32)[None, :].repeat(B, axis=0)

        alpha = alpha_full_column(logp, alpha_gamma=1.0)

        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)
        # First level should be ~0, last should be ~1
        np.testing.assert_allclose(alpha[:, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(alpha[:, -1], 1.0, atol=1e-5)

    def test_alpha_bottom_window(self):
        """Test bottom window alpha weighting."""
        B, N = 5, 30
        K = 10
        logp = np.linspace(2, 5, N, dtype=np.float32)[None, :].repeat(B, axis=0)

        alpha = alpha_bottom_window(logp, alpha_gamma=1.0, bot_window_k=K)

        # Top N-K levels should be zero
        np.testing.assert_allclose(alpha[:, :N-K], 0.0, atol=1e-5)

        # Bottom K levels should be non-zero
        assert np.all(alpha[:, -K:] > 0.0)

        # Last level should be 1
        np.testing.assert_allclose(alpha[:, -1], 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
