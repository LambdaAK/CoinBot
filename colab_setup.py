#!/usr/bin/env python3
"""
Colab Setup for DQN Training
Run this in Google Colab to train the DQN agent with GPU acceleration
"""

import os
import requests
import zipfile

def setup_colab():
    """Setup the environment for Colab training"""
    print("🚀 Setting up Colab environment for DQN training...")
    
    # Install required packages
    os.system("pip install torch torchvision torchaudio")
    os.system("pip install matplotlib numpy")
    
    # Create project directory
    os.makedirs("grid_world_dqn", exist_ok=True)
    os.chdir("grid_world_dqn")
    
    print("✅ Colab environment ready!")
    print("📁 Upload your files (grid_world.py, dqn_agent.py) to this directory")
    print("🤖 Then run: python dqn_agent.py")

if __name__ == "__main__":
    setup_colab() 