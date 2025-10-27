import torch
import pytest
from compliance_test import check_op_compliance, make_discontiguous_transpose

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)

@pytest.fixture(scope="session", autouse=True)
def print_env_info():
    print("")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

# if this fails, all of the other test results are unusable.
def test_discontiguous_creation():
    """Verify that make_discontiguous_transpose actually creates discontiguous tensors"""
    # Test 2D tensor
    t2d = torch.randn(4, 5)
    discontig_2d = make_discontiguous_transpose(t2d)
    assert not discontig_2d.is_contiguous(), "2D tensor should be discontiguous"
    assert torch.equal(t2d, discontig_2d), "Values should be unchanged"
    assert discontig_2d.shape == t2d.shape, "Shape should be preserved"

    # Test 3D tensor
    t3d = torch.randn(3, 4, 5)
    discontig_3d = make_discontiguous_transpose(t3d)
    assert not discontig_3d.is_contiguous(), "3D tensor should be discontiguous"
    assert torch.equal(t3d, discontig_3d), "Values should be unchanged"
    assert discontig_3d.shape == t3d.shape, "Shape should be preserved"

def test_linear():
    weight = torch.randn(12, 8)
    input_tensor = torch.randn(4, 8)

    results = check_op_compliance(
        torch.nn.functional.linear,
        input_tensor,
        weight,
        backend='mps'
    )

    for r in results:
        assert r.passed, r.message

def test_matmul():
    a = torch.randn(8, 8)
    b = torch.randn(8, 6)

    results = check_op_compliance(
        torch.matmul,
        a,
        b,
        backend='mps'
    )

    for r in results:
        assert r.passed, r.message

def test_bmm():
    a = torch.randn(4, 4, 8)
    b = torch.randn(4, 8, 6)

    results = check_op_compliance(
        torch.bmm,
        a,
        b,
        backend='mps'
    )

    for r in results:
        assert r.passed, r.message
