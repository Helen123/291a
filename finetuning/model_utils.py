import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import AutoModelForCausalLMWithValueHead
from typing import Tuple, Optional
import logging
import os

from config import ModelConfig

logger = logging.getLogger(__name__)


def create_bnb_config(model_config: ModelConfig) -> BitsAndBytesConfig:
    """
    Create BitsAndBytesConfig for 4-bit quantization.
    
    Args:
        model_config: Model configuration
        
    Returns:
        BitsAndBytesConfig object
    """
    return BitsAndBytesConfig(
        load_in_4bit=model_config.use_4bit,
        bnb_4bit_compute_dtype=getattr(torch, model_config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.use_nested_quant,
    )


def create_lora_config(model_config: ModelConfig) -> LoraConfig:
    """
    Create LoRA configuration.
    
    Args:
        model_config: Model configuration
        
    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=model_config.lora_target_modules,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_config: ModelConfig) -> AutoTokenizer:
    """
    Load and configure tokenizer.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="left"
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_base_model(model_config: ModelConfig) -> AutoModelForCausalLM:
    """
    Load base model with quantization.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Loaded model
    """
    # Create quantization config
    bnb_config = create_bnb_config(model_config)
    
    # Get current device (respects CUDA_VISIBLE_DEVICES)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_map = {"": 0}  # Use first visible device
    else:
        device_map = {"": "cuda:0"}  # Default to cuda:0
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
        use_cache=False,  # Disable cache for training
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model


def setup_lora_model(model: AutoModelForCausalLM, model_config: ModelConfig) -> AutoModelForCausalLM:
    """
    Setup LoRA on the base model.
    
    Args:
        model: Base model
        model_config: Model configuration
        
    Returns:
        Model with LoRA adapters
    """
    lora_config = create_lora_config(model_config)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_model_for_rl(model_config: ModelConfig, for_ppo: bool = True) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
    """
    Load model and tokenizer for RL training (PPO/GRPO).
    
    Args:
        model_config: Model configuration
        for_ppo: Whether this is for PPO training (needs value head)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = load_tokenizer(model_config)
    
    # Create quantization config
    bnb_config = create_bnb_config(model_config)
    
    # Load model with value head for RL
    if for_ppo:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16,
            peft_config=create_lora_config(model_config),
        )
    else:
        # For GRPO, load base model and add LoRA
        base_model = load_base_model(model_config)
        model = setup_lora_model(base_model, model_config)
    
    # Make sure model vocab size matches tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def load_sft_model(model_config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer for supervised fine-tuning.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = load_tokenizer(model_config)
    
    # Load and setup model
    model = load_base_model(model_config)
    model = setup_lora_model(model, model_config)
    
    return model, tokenizer


def save_model_and_tokenizer(model, tokenizer, save_path: str, push_to_hub: bool = False, 
                           hub_model_id: Optional[str] = None, hub_token: Optional[str] = None):
    """
    Save model and tokenizer locally and optionally push to HuggingFace Hub.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        save_path: Local path to save
        push_to_hub: Whether to push to Hub
        hub_model_id: Hub model ID
        hub_token: Hub token for authentication
    """
    logger.info(f"Saving model and tokenizer to {save_path}")
    
    # Save locally
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_path)
    else:
        # For PEFT models
        model.save_pretrained(save_path)
    
    tokenizer.save_pretrained(save_path)
    
    # Push to hub if requested
    if push_to_hub and hub_model_id:
        logger.info(f"Pushing model to HuggingFace Hub: {hub_model_id}")
        
        try:
            if hasattr(model, 'push_to_hub'):
                model.push_to_hub(
                    hub_model_id,
                    token=hub_token,
                    private=False
                )
            
            tokenizer.push_to_hub(
                hub_model_id,
                token=hub_token,
                private=False
            )
            
            logger.info("Successfully pushed to HuggingFace Hub")
            
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")


def calculate_model_size(model):
    """Calculate and log model size information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params
    } 