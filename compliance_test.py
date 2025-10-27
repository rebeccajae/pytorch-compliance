import torch
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class TestResult:
    op: str
    passed: bool
    message: str

def make_tensor_discontiguous(t: torch.Tensor) -> torch.Tensor:
    """Make tensor discontiguous via transpose and then transpose again, while preserving dimensions"""
    if t.dim() == 1:
         # skip, you're probably not striding here.
        return t
    elif t.dim() == 2:
        # Transpose and back seems to create discontiguous tensors
        return t.t().contiguous().t()
    elif t.dim() >= 3:
        # For 3D+ tensors, transpose last two dims
        return t.transpose(-2, -1).contiguous().transpose(-2, -1)
    else:
        # Yes I would like a 0-dimensional tensor.
        # Give me a negative two dimensional matrix.
        return t

def create_noncontiguous_tensor(shape, value=5.0, device='cpu', dtype=None):
    """Create a non-contiguous tensor with specified shape and value"""
    if dtype is None:
        dtype = torch.float32

    tensor = torch.full(shape, value, device=device, dtype=dtype)
    tensor = make_tensor_discontiguous(tensor)

    assert not tensor.is_contiguous(), "Expected non-contiguous tensor"
    return tensor

def check_op_compliance(op_fn: Callable, *args, backend='mps', atol=1e-4) -> List[TestResult]:
    """Check op with discontiguous tensors"""
    results = []

    # Move all args to backend first
    args_on_backend = tuple(
        arg.to(backend) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )

    for i, arg in enumerate(args_on_backend):
        if not isinstance(arg, torch.Tensor):
            continue

        # Setup discontiguous variant via transpose
        discontig = make_tensor_discontiguous(arg)
        contig = discontig.contiguous()

        # Verify it's actually discontiguous
        if discontig.is_contiguous():
            msg = f"arg[{i}] (shape={list(arg.shape)}): Could not make discontiguous, skipping"
            results.append(TestResult(op_fn.__name__, passed=True, message=msg))
            continue

        # Backend consistency check (discontiguous vs contiguous)
        test_args = list(args_on_backend)
        test_args[i] = discontig
        result_discontig = op_fn(*test_args)

        test_args[i] = contig
        result_contig = op_fn(*test_args)

        backend_ok = torch.allclose(result_discontig, result_contig, atol=atol)

        # CPU parity check
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in test_args]
        cpu_args[i] = discontig.cpu()  # Use discontiguous variant
        result_cpu = op_fn(*cpu_args)
        cpu_ok = torch.allclose(result_discontig.cpu(), result_cpu, atol=atol)

        passed = backend_ok and cpu_ok
        msg = f"arg[{i}] (shape={list(arg.shape)}): backend_consistent={backend_ok}, cpu_parity={cpu_ok}"

        results.append(TestResult(op_fn.__name__, passed, msg))

    return results

def check_inplace_op_with_noncontiguous_output(
    op_name: str,
    output_shape,
    output_value,
    *input_args,
    backend='mps',
    atol=1e-4,
    **op_kwargs
) -> TestResult:
    """Test an in-place operation with non-contiguous output tensor"""
    # Create non-contiguous outputs
    output_cpu = create_noncontiguous_tensor(
        output_shape, output_value, device='cpu'
    )
    output_backend = create_noncontiguous_tensor(
        output_shape, output_value, device=backend
    )

    # Move inputs to appropriate devices
    inputs_cpu = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in input_args]
    inputs_backend = [arg.to(backend) if isinstance(arg, torch.Tensor) else arg for arg in input_args]

    # Store initial values to detect silent failure
    initial_cpu = output_cpu.clone()
    initial_backend = output_backend.clone()

    # Run operations by calling the method on the tensor
    # This is a bit janky buuut... it works, I guess?
    getattr(output_cpu, op_name)(*inputs_cpu, **op_kwargs)
    getattr(output_backend, op_name)(*inputs_backend, **op_kwargs)

    # Check if CPU actually modified the tensor to validate that the CPU impl
    # works.
    cpu_changed = not torch.equal(initial_cpu, output_cpu)
    if not cpu_changed:
        return TestResult(
            op=op_name,
            passed=False,
            message="CPU did not modify non-contiguous output. Weird."
        )

    # Check if backend actually modified the tensor.
    backend_changed = not torch.equal(initial_backend, output_backend)
    if not backend_changed:
        return TestResult(
            op=op_name,
            passed=False,
            message=f"{backend} did not modify non-contiguous output"
        )

    # Check if results match
    matches = torch.allclose(output_cpu, output_backend.cpu(), atol=atol)
    if not matches:
        max_diff = (output_cpu - output_backend.cpu()).abs().max().item()
        return TestResult(
            op=op_name,
            passed=False,
            message=f"{backend} result differs from CPU (max diff: {max_diff:.6f})"
        )

    return TestResult(
        op=op_name,
        passed=True,
        message=f"{backend} matches CPU with non-contiguous output"
    )
