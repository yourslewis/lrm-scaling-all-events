#!/usr/bin/env python3
"""
GPU Keep-Alive Script

This script performs lightweight GPU operations every 12 hours to prevent 
GPU deallocation in cloud environments or to keep GPU drivers active.

Usage:
    python keepalive.py [--interval HOURS] [--device DEVICE_ID]

Options:
    --interval: Interval between GPU operations in hours (default: 12)
    --device: GPU device ID to use (default: 0, use 'auto' for automatic selection)
    --log-file: Path to log file (default: keepalive.log)
    --verbose: Enable verbose logging
"""

import argparse
import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional

import torch
import torch.nn as nn


class GPUKeepalive:
    """Manages GPU keep-alive operations."""
    
    def __init__(self, device_id: Optional[int] = None, log_file: str = "keepalive.log", verbose: bool = False):
        """
        Initialize the GPU keepalive manager.
        
        Args:
            device_id: GPU device ID to use. If None, auto-select.
            log_file: Path to log file.
            verbose: Enable verbose logging.
        """
        self.device_id = device_id
        self.log_file = log_file
        self.verbose = verbose
        self.running = True
        
        # Setup logging
        self._setup_logging()
        
        # Setup GPU device
        self.device = self._setup_device()
        
        # Create a simple model for GPU operations
        self.model = self._create_simple_model()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"GPU Keepalive initialized on device: {self.device}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create logger
        self.logger = logging.getLogger('gpu_keepalive')
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_device(self) -> torch.device:
        """Setup and validate GPU device."""
        if not torch.cuda.is_available():
            self.logger.error("CUDA is not available. This script requires a GPU.")
            sys.exit(1)
        
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"Found {num_gpus} GPU(s) available")
        
        if self.device_id is None:
            # Auto-select device (use device 0)
            device_id = 0
        else:
            device_id = self.device_id
        
        if device_id >= num_gpus:
            self.logger.error(f"Device ID {device_id} not available. Available devices: 0-{num_gpus-1}")
            sys.exit(1)
        
        device = torch.device(f'cuda:{device_id}')
        
        # Log GPU information
        gpu_name = torch.cuda.get_device_name(device_id)
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        self.logger.info(f"Using GPU {device_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return device
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple neural network for GPU operations."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(1024, 512)
                self.linear2 = nn.Linear(512, 256)
                self.linear3 = nn.Linear(256, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.relu(self.linear2(x))
                x = self.linear3(x)
                return x
        
        model = SimpleModel().to(self.device)
        return model
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def perform_gpu_operation(self):
        """Perform a lightweight GPU operation to keep GPU active."""
        try:
            start_time = time.time()
            
            # Create random input data
            batch_size = 64
            input_data = torch.randn(batch_size, 1024, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(input_data)
                
                # Perform some additional operations
                result = torch.mean(output)
                
                # Force GPU synchronization
                torch.cuda.synchronize()
            
            operation_time = time.time() - start_time
            
            # Log GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1e6  # MB
            memory_cached = torch.cuda.memory_reserved(self.device) / 1e6  # MB
            
            self.logger.info(
                f"GPU operation completed in {operation_time:.3f}s. "
                f"Result: {result:.6f}. "
                f"Memory: {memory_allocated:.1f}MB allocated, {memory_cached:.1f}MB cached"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"GPU operation failed: {str(e)}")
            return False
    
    def run(self, interval_hours: float = 12.0):
        """
        Run the keepalive loop.
        
        Args:
            interval_hours: Interval between GPU operations in hours.
        """
        interval_seconds = interval_hours * 3600
        
        self.logger.info(f"Starting GPU keepalive with {interval_hours} hour intervals")
        self.logger.info(f"Next operation scheduled at: {datetime.now() + timedelta(seconds=interval_seconds)}")
        
        # Perform initial operation
        self.perform_gpu_operation()
        
        while self.running:
            try:
                # Sleep in small chunks to allow for responsive shutdown
                sleep_chunks = int(interval_seconds / 10)  # 10 second chunks
                chunk_duration = interval_seconds / sleep_chunks
                
                for _ in range(sleep_chunks):
                    if not self.running:
                        break
                    time.sleep(chunk_duration)
                
                if self.running:
                    next_time = datetime.now() + timedelta(seconds=interval_seconds)
                    success = self.perform_gpu_operation()
                    
                    if success:
                        self.logger.info(f"Next operation scheduled at: {next_time}")
                    else:
                        self.logger.warning("GPU operation failed, but continuing...")
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt. Shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        self.logger.info("GPU keepalive stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GPU Keep-Alive Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=12.0,
        help='Interval between GPU operations in hours (default: 12.0)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='GPU device ID to use (default: auto, or specify device ID like 0, 1, etc.)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='keepalive.log',
        help='Path to log file (default: keepalive.log)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Parse device argument
    device_id = None
    if args.device != 'auto':
        try:
            device_id = int(args.device)
        except ValueError:
            print(f"Error: Invalid device ID '{args.device}'. Use 'auto' or a device number.")
            sys.exit(1)
    
    # Create and run keepalive
    try:
        keepalive = GPUKeepalive(
            device_id=device_id,
            log_file=args.log_file,
            verbose=args.verbose
        )
        keepalive.run(interval_hours=args.interval)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
