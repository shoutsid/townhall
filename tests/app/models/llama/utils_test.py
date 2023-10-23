import pytest
import numpy as np
from app.models.llama.utils import (
    precompute_freqs_cis,
    complex_mult,
    apply_rotary_emb
)
from tinygrad.tensor import Tensor

## precompute_freqs_cis

def test_precompute_freqs_cis_shape():
    dim = 6
    end = 5
    tensor = precompute_freqs_cis(dim, end)
    assert tensor.shape == (1, end, 1, dim // 2, 2), f"Shape mismatch: got {tensor.shape}, expected {(1, end, 1, dim // 2, 2)}"

def test_precompute_freqs_cis_cos_sin_values():
    dim = 6
    end = 5
    tensor = precompute_freqs_cis(dim, end).numpy()
    freqs = 1.0 / (10000.0 ** (np.arange(0, dim, 2)[:dim // 2] / dim))
    expected_freqs = np.arange(end)[:, np.newaxis] * freqs[np.newaxis, :]
    expected_cos = np.cos(expected_freqs)
    expected_sin = np.sin(expected_freqs)
    np.testing.assert_allclose(tensor[..., 0], expected_cos.reshape(1, end, 1, dim // 2), atol=1e-6)
    np.testing.assert_allclose(tensor[..., 1], expected_sin.reshape(1, end, 1, dim // 2), atol=1e-6)

def test_precompute_freqs_cis_theta():
    dim = 6
    end = 5
    theta = 20000.0
    tensor = precompute_freqs_cis(dim, end, theta=theta).numpy()
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[:dim // 2] / dim))
    expected_freqs = np.arange(end)[:, np.newaxis] * freqs[np.newaxis, :]
    expected_cos = np.cos(expected_freqs)
    np.testing.assert_allclose(tensor[..., 0], expected_cos.reshape(1, end, 1, dim // 2), atol=1e-6)

## complex_mult

def test_complex_mult_shape():
    A = Tensor(np.random.randn(2, 3, 4, 5, 2))
    c, d = 0.5, 0.3
    result = complex_mult(A, c, d)
    assert result.shape == A.shape, f"Shape mismatch: got {result.shape}, expected {A.shape}"

def test_complex_mult_correctness():
    A = Tensor(np.array([[[[[1, 2]]]]]))
    c, d = 3, 4
    result = complex_mult(A, c, d).numpy()
    expected_result = np.array([[[[[-5, 10]]]]])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)

def test_complex_mult_corner_cases():
    A = Tensor(np.array([[[[[0, 1]]]]]))
    c, d = 0, 1
    result = complex_mult(A, c, d).numpy()
    expected_result = np.array([[[[[-1, 0]]]]])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)

## rotary_emb

# Fixture for input tensor xq
@pytest.fixture
def xq_tensor():
    return Tensor(np.random.randn(2, 2, 4, 4))

# Fixture for input tensor xk
@pytest.fixture
def xk_tensor():
    return Tensor(np.random.randn(2, 3, 4, 4))

# Test for validating shape mismatch exception
def test_apply_rotary_emb_shape_mismatch(xk_tensor, xq_tensor):
    freqs_cis = Tensor(np.random.randn(2, 2, 4, 4, 2))
    with pytest.raises(AssertionError):
        apply_rotary_emb(xq_tensor, xk_tensor, freqs_cis)

# Test for checking rotary embeddings (mock a small example and test)
def test_apply_rotary_emb_results():
    xq_tensor = Tensor(np.array([[[[1, 2], [3, 4]]]]))
    xk_tensor = Tensor(np.array([[[[1, 2], [3, 4]]]]))
    freqs_cis = Tensor(np.array([[[[[1, 1], [1, 1]]]]]))
    xq_out, xk_out = apply_rotary_emb(xq_tensor, xk_tensor, freqs_cis)
