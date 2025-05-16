import sys

def check_device_and_packages(device):
    # Check torch is installed
    try:
        import torch
    except ImportError:
        print("torch is not installed. Please install it with `pip install torch`.")
        sys.exit(1)

    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA device requested but not available. Please check your CUDA installation or use 'cpu' or 'mps'.")
            sys.exit(1)
    elif device == "mps":
        if not torch.backends.mps.is_available() and not torch.backends.mps.is_built():
            print("MPS device requested but not available. Make sure you are on macOS with torch>=1.12 and a compatible Apple Silicon device.")
            sys.exit(1)
    elif device == "cpu":
        pass  # Always available
    else:
        print(f"Unknown device '{device}'. Please use 'cpu', 'cuda', or 'mps'.")
        sys.exit(1)
