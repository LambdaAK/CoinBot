#!/usr/bin/env python3
"""
Global Configuration for Grid World DQN Project
Centralized settings for CUDA, training parameters, and environment options
"""

import torch
import os

# ============================================================================
# CUDA CONFIGURATION
# ============================================================================

# Global CUDA settings
USE_CUDA = True  # Set to False to force CPU
FORCE_CPU = False  # Set to True to override CUDA detection
USE_MPS = False  # Set to False to disable MPS (Apple Silicon) - CPU is faster for small networks

# Auto-detect CUDA availability
def get_device():
    """Get the best available device (CUDA GPU, MPS, or CPU)"""
    if FORCE_CPU:
        return torch.device("cpu")
    
    # Check for CUDA first (NVIDIA GPUs)
    if USE_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for MPS (Apple Silicon GPUs)
    if USE_MPS and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    
    # Fallback to CPU
    return torch.device("cpu")

# Global device variable
DEVICE = get_device()

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# DQN Training Parameters
DQN_CONFIG = {
    'learning_rate': 0.0005,
    'gamma': 0.95,
    'epsilon': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.9995,
    'memory_size': 10000,
    'batch_size': 64,
    'target_update': 1000,
    'episodes': 1000000,
    'render_every': 10000,
    'save_every': 50000,
    'env_size': 5,
    'seed': 42
}

# Q-Learning Parameters
QL_CONFIG = {
    'learning_rate': 0.1,
    'discount_factor': 0.95,
    'epsilon': 0.1,
    'episodes': 100000,
    'render_every': 1000,
    'save_every': 10000,
    'env_size': 5,
    'seed': 42
}

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ENV_CONFIG = {
    'max_steps': 25,
    'obstacle_probability': 0.1,
    'use_dfs_validation': True
}

# ============================================================================
# COLAB OPTIMIZATION
# ============================================================================

# Colab-specific optimizations
COLAB_CONFIG = {
    'enable_progress_bars': True,
    'show_training_plots': True,
    'save_plots': True,
    'log_to_file': False,
    'use_tqdm': True  # Progress bars
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_device_info():
    """Print current device information"""
    print(f"🔧 Device Configuration:")
    print(f"   Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif DEVICE.type == 'mps':
        print(f"   GPU: Apple Silicon (MPS)")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   MPS Available: {torch.backends.mps.is_available()}")
    print(f"   PyTorch Version: {torch.__version__}")
    print()

def setup_colab_environment():
    """Setup optimized environment for Google Colab"""
    print("🚀 Setting up Colab environment...")
    
    # Install required packages
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    os.system("pip install matplotlib numpy tqdm")
    
    # Create directories
    os.makedirs("agents", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print_device_info()
    print("✅ Colab environment ready!")

def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print_device_info()
    
    if is_colab():
        print("🎯 Running in Google Colab - GPU acceleration enabled!")
        setup_colab_environment()
    else:
        print("💻 Running locally")
        print_device_info() 