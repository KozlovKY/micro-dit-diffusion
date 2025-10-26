#!/usr/bin/env python3
"""
Test script to verify callbacks are working correctly.
"""

import sys
import os
sys.path.append('/home/jovyan/Kozlov_KY/micro_diffusion')

from micro_diffusion.models.callbacks import TestCallback, DualModelSaver, TargetModelCheckpointSaver
from composer import State, Logger
from unittest.mock import Mock
import torch

def test_callbacks():
    print("Testing callbacks...")
    
    # Create mock objects
    mock_logger = Mock(spec=Logger)
    mock_model = Mock()
    mock_model.module = mock_model  # Simulate non-DDP model
    
    # Mock state
    mock_state = Mock(spec=State)
    mock_state.timestamp.batch.value = 50
    mock_state.model = mock_model
    
    # Test TestCallback
    print("\n1. Testing TestCallback...")
    test_callback = TestCallback()
    test_callback.batch_end(mock_state, mock_logger)
    
    # Test DualModelSaver
    print("\n2. Testing DualModelSaver...")
    dual_saver = DualModelSaver(save_folder="./test_ema", save_interval=50)
    
    # Mock consistency_loss_fn with target_model
    mock_target_model = Mock()
    mock_target_model.state_dict.return_value = {"param1": torch.tensor([1.0])}
    mock_model.consistency_loss_fn = Mock()
    mock_model.consistency_loss_fn.target_model = mock_target_model
    
    dual_saver.batch_end(mock_state, mock_logger)
    
    # Test TargetModelCheckpointSaver
    print("\n3. Testing TargetModelCheckpointSaver...")
    target_saver = TargetModelCheckpointSaver(
        save_folder="./test_target", 
        save_interval=50, 
        num_checkpoints_to_keep=2
    )
    
    target_saver.batch_end(mock_state, mock_logger)
    
    print("\nAll callback tests completed!")

if __name__ == "__main__":
    test_callbacks()
