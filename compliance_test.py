import torch
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class TestResult:
    op: str
    passed: bool
    message: str

def make_discontiguous_transpose(t: torch.Tensor) -> torch.Tensor:
    """Make tensor discontiguous while preserving dimensions"""
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
        discontig = make_discontiguous_transpose(arg)
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
