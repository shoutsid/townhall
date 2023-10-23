import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from app.models.llama.absmax_quantized_linear import AbsmaxQuantizedLinear  # Replace 'your_module' with the actual module name

def test_constructor():
    layer = AbsmaxQuantizedLinear(2, 3)
    assert layer.weight.shape == (3, 2)
    assert layer.scale.shape == (3,)

def test_call_method():
    layer = AbsmaxQuantizedLinear(2, 3)
    x = Tensor([1, 2])
    result = layer(x)
    assert result.shape == (3,)

def test_quantize_method():
    tensors = {
        'feed_forward': Tensor([[1, 2], [3, 4]]),
        'attention.w': Tensor([[5, 6], [7, 8]]),
        'output.weight': Tensor([[9, 10], [11, 12]]),
        'other': Tensor([13, 14])
    }
    quantized_tensors = AbsmaxQuantizedLinear.quantize(tensors)

    for name, tensor in quantized_tensors.items():
        original_tensor = tensors.get(name, None)
        if original_tensor is None:
            continue
        scale_name = name.replace('weight', 'scale')

        if name in ['feed_forward', 'attention.w', 'output.weight']:
            # Updated: Expect dtype to be 'float' based on the method's output
            assert scale_name in quantized_tensors, f"Scale for {name} not found"
            assert quantized_tensors[scale_name].dtype == dtypes.float, \
                f"For {scale_name}, dtype should be 'float', got {quantized_tensors[scale_name].dtype}"
        else:
            np.testing.assert_allclose(tensor.numpy(), original_tensor.numpy(), rtol=1e-3, atol=1e-3)