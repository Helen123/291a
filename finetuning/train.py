#!/usr/bin/env python3
"""
Main training script for CodeLlama fine-tuning with QLoRA.
Supports both PPO and GRPO training methods.
"""

import os
import argparse
import logging
import torch
from pathlib import Path
import wandb
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import ExperimentConfig, load_config_from_env
from ppo_trainer import CodePPOTrainer
from grpo_trainer import CodeGRPOTrainer
from model_utils import calculate_model_size

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment variables and paths."""
    # Set HuggingFace cache directory
    os.environ.setdefault("HF_HOME", "./cache")
    
    # Set transformers cache
    os.environ.setdefault("TRANSFORMERS_CACHE", "./cache/transformers")
    
    # Allow code execution for reward function
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    
    # Create necessary directories
    Path("./cache").mkdir(exist_ok=True)
    Path("./checkpoints").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune CodeLlama with QLoRA")
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["ppo", "grpo"],
        default="ppo",
        help="Training method to use (default: ppo)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="codellama/CodeLlama-7b-Python-hf",
        help="Model name or path"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID for uploading"
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final model to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="codellama-mbpp-finetuning",
        help="Weights & Biases project name"
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with smaller dataset"
    )
    
    return parser.parse_args()


def update_config_with_args(config: ExperimentConfig, args) -> ExperimentConfig:
    """Update configuration with command line arguments."""
    # Method
    config.method = args.method
    
    # Model configuration
    config.model.model_name = args.model_name
    config.model.max_seq_length = args.max_length
    config.model.lora_r = args.lora_r
    config.model.lora_alpha = args.lora_alpha
    
    # Training configuration
    config.training.per_device_train_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.learning_rate = args.learning_rate
    config.training.num_train_epochs = args.num_epochs
    config.training.save_steps = args.save_steps
    config.training.logging_steps = args.logging_steps
    
    # Output directory
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    # HuggingFace Hub
    if args.hub_model_id:
        config.hf.hub_model_id = args.hub_model_id
    config.hf.push_to_hub = args.push_to_hub
    
    # Wandb
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    
    # Seed
    config.seed = args.seed
    
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_config(config: ExperimentConfig):
    """Print configuration summary."""
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Method: {config.method.upper()}")
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Max sequence length: {config.model.max_seq_length}")
    logger.info(f"LoRA rank: {config.model.lora_r}")
    logger.info(f"LoRA alpha: {config.model.lora_alpha}")
    logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Epochs: {config.training.num_train_epochs}")
    logger.info(f"Output directory: {config.training.output_dir}")
    logger.info(f"Use Wandb: {config.use_wandb}")
    if config.use_wandb:
        logger.info(f"Wandb project: {config.wandb_project}")
        logger.info(f"Wandb run: {config.wandb_run_name}")
    logger.info(f"Push to Hub: {config.hf.push_to_hub}")
    if config.hf.push_to_hub:
        logger.info(f"Hub model ID: {config.hf.hub_model_id}")
    logger.info("=" * 60)


def merge_adapter_to_base(adapter_path: str, base_model_name: str, merged_output_dir: str) -> str:
    """
    Merge LoRA adapter to base model and save the merged model.
    
    Args:
        adapter_path: Path to the trained adapter
        base_model_name: Name/path of the base model
        merged_output_dir: Directory to save merged model
        
    Returns:
        Path to the merged model
    """
    logger.info(f"🔧 Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    logger.info(f"🔗 Loading adapter from: {adapter_path}")
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("🔄 Merging adapter to base model...")
    
    # Merge adapter weights to base model
    merged_model = model.merge_and_unload()
    
    logger.info(f"💾 Saving merged model to: {merged_output_dir}")
    
    # Create output directory
    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save merged model
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)
    
    logger.info("✅ Model merging completed successfully!")
    
    return merged_output_dir


def run_training_experiment(config: ExperimentConfig):
    """
    Run complete training experiment with automatic adapter merging.
    
    Args:
        config: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING CODELLAMA FINE-TUNING WITH BIGCODE-COMPATIBLE EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Training method: {config.training.method}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    logger.info(f"Max epochs: {config.training.max_epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    
    # Initialize WandB if enabled
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.experiment_name,
            config=config.to_dict()
        )
        logger.info(f"📊 WandB logging initialized: {config.logging.wandb_project}")
    
    try:
        # Initialize trainer based on method
        if config.training.method.upper() == "PPO":
            logger.info("🏃 Initializing PPO trainer...")
            trainer = CodePPOTrainer(config)
        elif config.training.method.upper() == "GRPO":
            logger.info("🏃 Initializing GRPO trainer...")
            trainer = CodeGRPOTrainer(config)
        else:
            raise ValueError(f"Unknown training method: {config.training.method}")
        
        # Setup trainer components
        logger.info("🔧 Setting up model and data...")
        trainer.setup_model()
        trainer.setup_data()
        trainer.setup_reward_function()
        
        # Start training
        logger.info("🎯 Starting training...")
        training_results = trainer.train()
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Log final training metrics
        if training_results and isinstance(training_results, dict):
            for key, value in training_results.items():
                logger.info(f"{key}: {value}")
        
        # Get adapter path (should be saved in output directory)
        adapter_path = Path(config.paths.output_dir)
        
        # Check if adapter was saved
        if not (adapter_path / "adapter_model.safetensors").exists():
            logger.warning("⚠️ No adapter found, checking for alternative save formats...")
            if not any(adapter_path.glob("*.safetensors")) and not any(adapter_path.glob("*.bin")):
                logger.error("❌ No model files found in output directory")
                return
        
        # AUTO-MERGE ADAPTER TO BASE MODEL
        logger.info("🔗 AUTO-MERGING ADAPTER TO BASE MODEL...")
        
        merged_model_dir = str(adapter_path.parent / f"{adapter_path.name}_merged")
        
        try:
            merged_path = merge_adapter_to_base(
                adapter_path=str(adapter_path),
                base_model_name=config.model.model_name,
                merged_output_dir=merged_model_dir
            )
            
            logger.info("=" * 80)
            logger.info("🎯 EXPERIMENT COMPLETED WITH AUTO-MERGE!")
            logger.info("=" * 80)
            logger.info(f"📁 Adapter saved at: {adapter_path}")
            logger.info(f"📁 Merged model saved at: {merged_path}")
            
            # Log evaluation command
            logger.info("\n📋 BIGCODE-COMPATIBLE EVALUATION COMMANDS:")
            logger.info("-" * 50)
            
            # Evaluation with our compatible script
            logger.info("1. Using our bigcode-compatible evaluator:")
            logger.info(f"python finetuning/evaluate.py {merged_path} --num_samples 15 --temperature 0.1")
            
            # Evaluation with official bigcode-evaluation-harness
            logger.info("\n2. Using official bigcode-evaluation-harness:")
            logger.info(f"""cd bigcode-evaluation-harness
accelerate launch main.py \\
  --model {merged_path} \\
  --max_length_generation 512 \\
  --tasks mbpp \\
  --temperature 0.1 \\
  --n_samples 15 \\
  --batch_size 10 \\
  --allow_code_execution""")
            
        except Exception as e:
            logger.error(f"❌ Failed to merge adapter: {e}")
            logger.info("🔧 You can manually merge later using:")
            logger.info(f"python finetuning/merge_adapter.py --adapter_model_id {adapter_path} --output_dir {merged_model_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("Please check the logs above for details")
        if config.logging.use_wandb:
            wandb.finish(exit_code=1)
        raise
    
    finally:
        # Clean up WandB
        if config.logging.use_wandb:
            wandb.finish()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file>")
        print("Example: python train.py configs/ppo_config.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration
    try:
        config = ExperimentConfig.from_yaml(config_file)
        logger.info(f"✅ Configuration loaded from: {config_file}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Set environment variables for code execution
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Create output directory
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    try:
        run_training_experiment(config)
        logger.info("🎉 Full experiment pipeline completed successfully!")
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"💥 Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 