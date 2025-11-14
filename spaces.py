# Mock spaces module for non-HuggingFace Spaces environments (e.g., RunPod)
# This provides a no-op decorator that replaces @spaces.GPU

def GPU(*args, **kwargs):
    """
    Mock GPU decorator that does nothing.
    On HuggingFace Spaces, this decorator manages GPU allocation.
    On RunPod/local, GPUs are always available, so this is a no-op.
    """
    def decorator(func):
        return func
    # Handle both @GPU and @GPU() syntax
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator
