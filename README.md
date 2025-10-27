# `pytorch-compliance` - An investigatory correctness-check for MPS
This is a repository where I'm going to put some compliance-checking tool for
MPS PyTorch.

I previously ran into an issue with PyTorch's correctness with discontiguous
tensors in https://github.com/pytorch/pytorch/issues/161640. A friend of mine
mentioned that someone else ran into a [similar issue][1].

I figured I should catalog these bugs somewhere, and maybe make tools to
improve checking compliance.

## What's it do?
Simple, it runs the same operation on CPU and some accelerator. It does so
with both contiguous and discontiguous inputs. It then compares the values
of the operation. In practice they should be _pretty close_ but may not be
perfectly identical.

## Usage
- Put test in `test_ops.py`
- `uv run pytest -v -s test_ops.py`.

[1]: https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/
