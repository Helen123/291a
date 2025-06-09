import torch
import torch.nn.functional as F
import logging
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from trl import GRPOConfig, GRPOTrainer
import math
from tqdm import tqdm

from config import ExperimentConfig
from model_utils import load_tokenizer, load_base_model, setup_lora_model, save_model_and_tokenizer, calculate_model_size
from data_utils import create_mbpp_dataset_for_rl, RewardFunction, MBPPDataProcessor, create_dataset_for_rl

logger = logging.getLogger(__name__)


class CodeGRPOTrainer:
    """Standard GRPO Trainer for code generation with Q-LoRA."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize GRPO trainer.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.reward_fn = None
        self.dataset = None
        self.optimizer = None
        
        # GRPO hyperparameters - optimized stable settings
        self.learning_rate = config.training.learning_rate or 3e-6  # Increase learning rate to reasonable level
        self.num_samples = getattr(config.training, 'num_samples', 8)  # Increase number of samples
        self.clip_range = getattr(config.training, 'clip_range', 0.2)  # Standard PPO clipping
        self.kl_coeff = getattr(config.training, 'kl_coeff', 0.02)  # Reduce KL penalty to allow more exploration
        self.entropy_coeff = getattr(config.training, 'entropy_coeff', 0.01)  # Encourage exploration
        self.value_coeff = getattr(config.training, 'value_coeff', 0.5)  # Value function coefficient
        
        # Advantage normalization - enable whitening to reduce variance
        self.use_advantage_whitening = getattr(config.training, 'use_advantage_whitening', True)  # Enable whitening
        self.advantage_norm_eps = getattr(config.training, 'advantage_norm_eps', 1e-8)
        
        # Temperature scheduling - dynamic temperature scheduling
        self.temperature_schedule = getattr(config.training, 'temperature_schedule', True)  # Enable temperature scheduling
        self.initial_temperature = getattr(config.training, 'initial_temperature', 1.0)  # Initial temperature
        self.min_temperature = getattr(config.training, 'min_temperature', 0.7)  # Minimum temperature
        
        # Early stopping and best model tracking
        self.best_test_pass_rate = 0.0
        self.best_reward = 0.0
        # Get patience from config, default to 2, support disabling early stopping by setting to 0
        self.patience = getattr(config.training, 'early_stopping_patience', 2)
        # If early stopping is explicitly disabled, set patience to 0
        if getattr(config, 'no_early_stopping', False):
            self.patience = 0
        self.patience_counter = 0
        self.early_stop = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def setup_model(self):
        """Setup model and tokenizer for GRPO training."""
        logger.info("Loading model and tokenizer for GRPO training...")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(self.config.model)
        
        # Fix attention mask issue by setting a proper pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        
        # Ensure pad_token is different from eos_token
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        # Load base model and setup LoRA for policy model
        base_model = load_base_model(self.config.model)
        base_model = setup_lora_model(base_model, self.config.model)
        
        # Resize token embeddings if we added new tokens
        if len(self.tokenizer) > base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(self.tokenizer))
        
        self.model = base_model
        
        # Create reference model (frozen copy)
        ref_base_model = load_base_model(self.config.model)
        ref_base_model = setup_lora_model(ref_base_model, self.config.model)
        
        # Resize reference model embeddings too
        if len(self.tokenizer) > ref_base_model.config.vocab_size:
            ref_base_model.resize_token_embeddings(len(self.tokenizer))
            
        self.ref_model = ref_base_model.eval()
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.config.training.weight_decay
        )
        
        # Calculate and log model size
        model_stats = calculate_model_size(self.model)
        logger.info(f"Policy model loaded with {model_stats['trainable_params']:,} trainable parameters")
        logger.info("Reference model created (frozen copy)")
        logger.info(f"GRPO hyperparameters:")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Number of samples per prompt: {self.num_samples}")
        logger.info(f"  Clipping parameter ε: {self.clip_range}")
        logger.info(f"  KL divergence coefficient β: {self.kl_coeff}")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        
    def setup_data(self):
        """Setup training dataset."""
        dataset_name = self.config.data.dataset_name.lower()
        logger.info(f"Loading and preprocessing {dataset_name.upper()} dataset...")
        
        # Load dataset based on configuration
        self.dataset = create_dataset_for_rl(
            dataset_name=dataset_name,
            split=self.config.data.split,
            max_samples=self.config.data.max_samples
        )
        
        # Filter out examples that are too long
        def filter_long_examples(example):
            prompt_length = len(self.tokenizer.encode(example["prompt"]))
            return prompt_length <= self.config.data.max_prompt_length
        
        self.dataset = self.dataset.filter(filter_long_examples)
        
        logger.info(f"{dataset_name.upper()} dataset loaded with {len(self.dataset)} examples")
        
    def setup_reward_function(self):
        """Setup reward function for code execution."""
        dataset_name = self.config.data.dataset_name.lower()
        logger.info(f"Setting up reward function for {dataset_name.upper()} dataset...")
        
        self.reward_fn = RewardFunction(
            timeout=self.config.training.code_execution_timeout,
            max_reward=1.0,
            min_reward=0.0,
            syntax_penalty=-0.1,
            dataset_type=dataset_name
        )
        
    def generate_responses(self, prompts: List[str], epoch: int = 0) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
        """
        Generate multiple responses for each prompt with dynamic temperature.
        
        Args:
            prompts: List of input prompts
            epoch: Current training epoch for temperature scheduling
            
        Returns:
            Tuple of (responses, log_probs) for each prompt
        """
        all_responses = []
        all_log_probs = []
        
        # Dynamic temperature adjustment - more stable scheduling
        if self.temperature_schedule:
            # Linear decay instead of exponential decay, more stable
            progress = epoch / max(1, self.config.training.num_train_epochs - 1)
            current_temp = self.initial_temperature * (1 - progress) + self.min_temperature * progress
            current_temp = max(self.min_temperature, current_temp)
        else:
            current_temp = self.config.training.temperature
        
        device = next(self.model.parameters()).device
        
        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                prompt_tokens = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.config.data.max_prompt_length
                ).to(device)
                
                responses = []
                log_probs_list = []
                
                for _ in range(self.num_samples):
                    # Generate response with dynamic temperature
                    outputs = self.model.generate(
                        prompt_tokens['input_ids'],
                        attention_mask=prompt_tokens['attention_mask'],
                        max_new_tokens=self.config.training.max_new_tokens,
                        do_sample=self.config.training.do_sample,
                        temperature=current_temp,  # Use dynamic temperature
                        top_p=self.config.training.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    # Extract generated sequence
                    generated_ids = outputs.sequences[0]
                    response_ids = generated_ids[prompt_tokens['input_ids'].shape[1]:]
                    
                    # Decode response
                    response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    responses.append(response)
                    
                    # Compute log probabilities for the generated sequence
                    full_sequence = generated_ids.unsqueeze(0)
                    attention_mask = torch.ones_like(full_sequence)
                    
                    # Get log probabilities from current model
                    model_outputs = self.model(full_sequence, attention_mask=attention_mask)
                    logits = model_outputs.logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get log probabilities for response tokens only
                    response_start = prompt_tokens['input_ids'].shape[1]
                    response_log_probs = log_probs[0, response_start-1:-1, :]
                    response_tokens = full_sequence[0, response_start:]
                    
                    # Select log probs for actual tokens
                    selected_log_probs = response_log_probs.gather(
                        1, response_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    log_probs_list.append(selected_log_probs)
                
                all_responses.append(responses)
                all_log_probs.append(log_probs_list)
        
        return all_responses, all_log_probs
    
    def compute_reference_log_probs(self, prompts: List[str], responses: List[List[str]]) -> List[List[torch.Tensor]]:
        """
        Compute log probabilities from reference model.
        
        Args:
            prompts: List of input prompts
            responses: List of lists of responses
            
        Returns:
            List of lists of log probabilities from reference model
        """
        all_ref_log_probs = []
        device = next(self.ref_model.parameters()).device
        
        self.ref_model.eval()
        with torch.no_grad():
            for prompt, response_list in zip(prompts, responses):
                ref_log_probs_list = []
                
                for response in response_list:
                    # Combine prompt and response
                    full_text = prompt + response
                    
                    # Tokenize
                    tokens = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.model.max_seq_length
                    ).to(device)
                    
                    # Get reference model outputs
                    ref_outputs = self.ref_model(**tokens)
                    ref_logits = ref_outputs.logits
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    
                    # Get log probs for response tokens
                    prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                    response_start = prompt_length
                    response_end = tokens['input_ids'].shape[1]
                    
                    if response_end > response_start:
                        response_log_probs = ref_log_probs[0, response_start-1:response_end-1, :]
                        response_tokens = tokens['input_ids'][0, response_start:response_end]
                        
                        selected_ref_log_probs = response_log_probs.gather(
                            1, response_tokens.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        selected_ref_log_probs = torch.tensor([], device=device)
                    
                    ref_log_probs_list.append(selected_ref_log_probs)
                
                all_ref_log_probs.append(ref_log_probs_list)
        
        return all_ref_log_probs
    
    def compute_grpo_loss(self, log_probs: List[List[torch.Tensor]], 
                         ref_log_probs: List[List[torch.Tensor]], 
                         rewards: List[List[float]]) -> torch.Tensor:
        """
        Compute GRPO loss with correct implementation.
        
        GRPO uses group-based relative policy optimization:
        1. For each group of samples, compute relative advantages
        2. Apply importance sampling with clipping
        3. Add KL divergence penalty
        
        Args:
            log_probs: Log probabilities from policy model
            ref_log_probs: Log probabilities from reference model  
            rewards: Reward values for each sample
            
        Returns:
            GRPO loss tensor
        """
        device = next(self.model.parameters()).device
        total_policy_loss = torch.tensor(0.0, device=device)
        total_kl_loss = torch.tensor(0.0, device=device)
        num_valid_groups = 0
        
        for prompt_log_probs, prompt_ref_log_probs, prompt_rewards in zip(log_probs, ref_log_probs, rewards):
            if len(prompt_rewards) < 2:
                continue  # Need at least 2 samples for group comparison
            
            # Step 1: Compute relative advantages within group
            rewards_tensor = torch.tensor(prompt_rewards, device=device, dtype=torch.float32)
            
            # Improved advantage calculation
            if len(prompt_rewards) >= 3:
                # Use standardized advantage calculation
                mean_reward = rewards_tensor.mean()
                std_reward = rewards_tensor.std() + 1e-8  # Prevent division by zero
                
                # Z-score normalization for advantages
                advantages = (rewards_tensor - mean_reward) / std_reward
                
                # Optional: use rank-based advantage as backup
                if std_reward < 1e-6:  # Use rank when rewards are basically the same
                    ranked_indices = torch.argsort(rewards_tensor, descending=True)
                    advantages = torch.zeros_like(rewards_tensor)
                    for i, idx in enumerate(ranked_indices):
                        advantages[idx] = 1.0 - 2.0 * i / (len(prompt_rewards) - 1)
            else:
                # Simple relative advantage for 2 samples
                mean_reward = rewards_tensor.mean()
                advantages = rewards_tensor - mean_reward
                # Normalize to prevent extreme values
                if advantages.std() > 1e-8:
                    advantages = advantages / (advantages.std() + 1e-8)
                advantages = torch.clamp(advantages, -2.0, 2.0)  # Wider clamp range
            
            # Advantage whitening (optional)
            if self.use_advantage_whitening and len(advantages) > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std() + self.advantage_norm_eps
                advantages = (advantages - adv_mean) / adv_std
            
            group_policy_loss = torch.tensor(0.0, device=device)
            group_kl_loss = torch.tensor(0.0, device=device)
            valid_samples = 0
            
            # Step 2: Compute loss for each sample in the group
            for i, (lp, ref_lp, adv) in enumerate(zip(prompt_log_probs, prompt_ref_log_probs, advantages)):
                if len(lp) == 0 or len(ref_lp) == 0:
                    continue
                
                # Ensure tensors have same length
                min_len = min(len(lp), len(ref_lp))
                if min_len == 0:
                    continue
                    
                lp = lp[:min_len]
                ref_lp = ref_lp[:min_len]
                
                # Step 3: Compute importance sampling ratio
                log_ratio = (lp - ref_lp).sum()  # log(π/π_ref)
                
                # Clamp for numerical stability
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                
                # Step 4: PPO-style clipped objective
                unclipped_obj = ratio * adv
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                clipped_obj = clipped_ratio * adv
                
                # Take minimum for conservative update
                policy_objective = torch.min(unclipped_obj, clipped_obj)
                group_policy_loss += policy_objective
                
                # Step 5: KL divergence (correct calculation)
                # KL(π||π_ref) = Σ π(x) * log(π(x)/π_ref(x))
                # For discrete case: KL ≈ log(π/π_ref) when π ≈ π_ref
                kl_penalty = torch.abs(log_ratio)  # Use absolute value for stability
                group_kl_loss += kl_penalty
                
                valid_samples += 1
            
            if valid_samples > 0:
                # Average over valid samples in group
                group_policy_loss = group_policy_loss / valid_samples
                group_kl_loss = group_kl_loss / valid_samples
                
                total_policy_loss += group_policy_loss
                total_kl_loss += group_kl_loss
                num_valid_groups += 1
        
        if num_valid_groups > 0:
            # Average over groups
            avg_policy_loss = total_policy_loss / num_valid_groups
            avg_kl_loss = total_kl_loss / num_valid_groups
            
            # GRPO objective: maximize policy objective - KL penalty
            total_loss = -avg_policy_loss + self.kl_coeff * avg_kl_loss
            
            # Add numerical stability check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("GRPO loss is NaN or Inf, using small positive loss")
                total_loss = torch.tensor(0.01, device=device)
                
            # Log metrics
            if hasattr(self, '_step_count'):
                self._step_count += 1
            else:
                self._step_count = 1
                
            if self._step_count % 5 == 0:
                logger.info(f"GRPO Metrics - Policy: {avg_policy_loss.item():.4f}, KL: {avg_kl_loss.item():.4f}, Total: {total_loss.item():.4f}")
        else:
            total_loss = torch.tensor(0.01, device=device)
        
        return total_loss
    
    def _save_merged_epoch_model(self, epoch_num: int, test_pass_rate: float, reward: float):
        """
        Save merged model for specified epoch and clean up original adapter weights
        
        Args:
            epoch_num: Epoch number
            test_pass_rate: Test pass rate
            reward: Average reward
        """
        # Import inside method to avoid circular imports
        from train import merge_adapter_to_base, extract_model_short_name
        from pathlib import Path
        import shutil
        
        logger.info(f"💾 Starting to save merged model for epoch {epoch_num}...")
        
        # First save current epoch's adapter
        epoch_adapter_path = f"{self.config.training.output_dir}/epoch_{epoch_num}_adapter"
        save_model_and_tokenizer(
            self.model, 
            self.tokenizer, 
            epoch_adapter_path,
            push_to_hub=False
        )
        
        try:
            # Generate merged model save path
            base_model_name = self.config.model.model_name
            dataset_name = self.config.data.dataset_name
            method = self.config.method
            model_short_name = extract_model_short_name(base_model_name)
            
            # Format: model-dataset-method-qlora-merged-epoch{N}-pass{pass_rate:.3f}
            merged_model_dir = (f"./checkpoints/{model_short_name}-{dataset_name}-{method}-qlora-merged-"
                              f"epoch{epoch_num}-pass{test_pass_rate:.3f}")
            
            # Execute merge
            logger.info(f"🔗 Merging adapter to base model...")
            merge_adapter_to_base(
                adapter_path=epoch_adapter_path,
                base_model_name=base_model_name,
                merged_output_dir=merged_model_dir
            )
            
            # Delete adapter weights after successful merge
            if Path(epoch_adapter_path).exists():
                logger.info(f"🗑️  Deleting original adapter weights: {epoch_adapter_path}")
                shutil.rmtree(epoch_adapter_path)
            
            logger.info(f"✅ Epoch {epoch_num} merge model saved successfully!")
            logger.info(f"📁 Save path: {merged_model_dir}")
            logger.info(f"📊 Test pass rate: {test_pass_rate:.3f}, Average reward: {reward:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Epoch {epoch_num} merge model save failed: {e}")
            logger.error("Retaining original adapter weights for manual processing")

    def train_step(self, batch_data) -> Dict[str, float]:
        """
        Perform one GRPO training step.
        
        Args:
            batch_data: Training batch
            
        Returns:
            Training statistics
        """
        prompts = batch_data["prompt"]
        test_cases = batch_data["test_cases"]
        
        # Step 1: Generate responses using current policy
        responses, policy_log_probs = self.generate_responses(prompts)
        
        # Step 2: Compute reference log probabilities (no gradients needed)
        ref_log_probs = self.compute_reference_log_probs(prompts, responses)
        
        # Step 3: Compute rewards
        all_rewards = []
        all_details = []
        
        for prompt, response_list, test_case in zip(prompts, responses, test_cases):
            # Combine prompts and responses for reward computation
            full_codes = [prompt + response for response in response_list]
            
            rewards, details = self.reward_fn.compute_batch_rewards_detailed(
                generated_codes=full_codes,
                prompts=[prompt] * len(response_list),
                test_cases=[test_case] * len(response_list),
                max_workers=4
            )
            
            all_rewards.append(rewards)
            all_details.extend(details)
        
        # Step 4: Recompute policy log probabilities while maintaining consistency
        device = next(self.model.parameters()).device
        corrected_policy_log_probs = []
        
        self.model.train()  # Ensure model is in training mode
        
        for prompt, response_list in zip(prompts, responses):
            prompt_log_probs = []
            
            for response in response_list:
                # Combine prompt and response
                full_text = prompt + response
                
                # Tokenize - ensure consistency with generation time
                tokens = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.model.max_seq_length,
                    padding=False,
                    add_special_tokens=True
                ).to(device)
                
                # Forward pass with gradients enabled
                outputs = self.model(**tokens)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get log probs for response tokens - more precise calculation
                prompt_tokens = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.config.data.max_prompt_length,
                    add_special_tokens=True
                ).to(device)
                
                prompt_length = prompt_tokens['input_ids'].shape[1]
                response_start = prompt_length
                response_end = tokens['input_ids'].shape[1]
                
                if response_end > response_start:
                    # Accurately get log probs for response part
                    response_log_probs = log_probs[0, response_start-1:response_end-1, :]
                    response_tokens = tokens['input_ids'][0, response_start:response_end]
                    
                    selected_log_probs = response_log_probs.gather(
                        1, response_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    selected_log_probs = torch.tensor([], device=device, requires_grad=True)
                
                prompt_log_probs.append(selected_log_probs)
            
            corrected_policy_log_probs.append(prompt_log_probs)
        
        # Step 5: Compute GRPO loss with gradients
        grpo_loss = self.compute_grpo_loss(corrected_policy_log_probs, ref_log_probs, all_rewards)
        
        # Step 6: Backward pass
        self.optimizer.zero_grad()
        grpo_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.max_grad_norm)
        
        self.optimizer.step()
        
        # Compute statistics
        flat_rewards = [r for reward_list in all_rewards for r in reward_list]
        
        # Compute metrics similar to PPO trainer
        valid_details = [d for d in all_details if d and not d.get('has_syntax_error', False)]
        
        if valid_details:
            avg_test_ratio = sum(d.get('test_ratio', 0.0) for d in valid_details) / len(valid_details)
            avg_quality_score = sum(d.get('quality_score', 0.0) for d in valid_details) / len(valid_details)
            total_passed = sum(d.get('passed_tests', 0) for d in all_details)
            total_tests = sum(d.get('total_tests', 0) for d in all_details)
        else:
            avg_test_ratio = 0.0
            avg_quality_score = 0.0
            total_passed = 0
            total_tests = 0
        
        syntax_errors = sum(1 for d in all_details if d.get('has_syntax_error', False))
        perfect_solutions = sum(1 for d in all_details if d.get('test_ratio', 0.0) == 1.0)
        
        stats = {
            "rewards/mean": np.mean(flat_rewards) if flat_rewards else 0.0,
            "rewards/max": np.max(flat_rewards) if flat_rewards else 0.0,
            "rewards/min": np.min(flat_rewards) if flat_rewards else 0.0,
            "success_rate": sum(1 for r in flat_rewards if r > 0) / len(flat_rewards) if flat_rewards else 0.0,
            "test_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            "avg_test_ratio": avg_test_ratio,
            "avg_quality_score": avg_quality_score,
            "pass_at_1": perfect_solutions / len(all_details) if all_details else 0.0,
            "syntax_errors": syntax_errors,
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "grpo_loss": grpo_loss.item(),
            "num_samples": len(flat_rewards),
        }
        
        return stats

    def train(self):
        """Run GRPO training."""
        logger.info("Starting standard GRPO training with Q-LoRA...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_reward_function()
        
        # Initialize wandb if enabled
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
        
        # Training loop
        total_steps = 0
        best_success_rate = 0.0
        
        # Create batches from dataset
        batch_size = self.config.training.per_device_train_batch_size
        num_batches = len(self.dataset) // batch_size
        
        logger.info(f"Starting GRPO training with {num_batches} batches per epoch")
        logger.info(f"Dataset size: {len(self.dataset)} samples")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Samples per prompt: {self.num_samples}")
        
        for epoch in range(self.config.training.num_train_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_train_epochs}")
            
            epoch_stats = []
            
            # Add progress bar for each epoch
            pbar = tqdm(
                range(num_batches), 
                desc=f"Epoch {epoch + 1}/{self.config.training.num_train_epochs}",
                unit="batch",
                ncols=140
            )
            
            for batch_idx in pbar:
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = self.dataset.select(range(start_idx, end_idx))
                
                # Perform GRPO training step
                try:
                    step_stats = self.train_step(batch_data)
                    epoch_stats.append(step_stats)
                    total_steps += 1
                    
                    # Update progress bar with current metrics
                    pbar.set_postfix({
                        'Reward': f"{step_stats.get('rewards/mean', 0.0):.3f}",
                        'Success': f"{step_stats.get('success_rate', 0.0):.2f}",
                        'Pass@1': f"{step_stats.get('pass_at_1', 0.0):.2f}",
                        'TestPass': f"{step_stats.get('test_pass_rate', 0.0):.2f}",
                        'GRPOLoss': f"{step_stats.get('grpo_loss', 0.0):.3f}",
                        'Samples': f"{step_stats.get('num_samples', 0)}"
                    })
                    
                    # Log statistics
                    if total_steps % self.config.training.logging_steps == 0:
                        logger.info(f"Step {total_steps}: {step_stats}")
                        
                        if self.config.use_wandb:
                            wandb.log(step_stats, step=total_steps)
                    
                    # Save checkpoint
                    if total_steps % self.config.training.save_steps == 0:
                        checkpoint_path = f"{self.config.training.output_dir}/checkpoint-{total_steps}"
                        save_model_and_tokenizer(
                            self.model, 
                            self.tokenizer, 
                            checkpoint_path,
                            push_to_hub=False
                        )
                        
                        # Save best model based on success rate
                        current_success_rate = step_stats.get("success_rate", 0.0)
                        if current_success_rate > best_success_rate:
                            best_success_rate = current_success_rate
                            best_model_path = f"{self.config.training.output_dir}/best_model"
                            save_model_and_tokenizer(
                                self.model, 
                                self.tokenizer, 
                                best_model_path,
                                push_to_hub=False
                            )
                            logger.info(f"New best model saved with success rate: {best_success_rate:.4f}")
                
                except Exception as e:
                    logger.error(f"Error in training step {total_steps}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Close progress bar and log epoch summary
            pbar.close()
            
            # Calculate epoch summary statistics
            if epoch_stats:
                avg_reward = sum(s.get('rewards/mean', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_success_rate = sum(s.get('success_rate', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_test_pass_rate = sum(s.get('test_pass_rate', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_pass_at_1 = sum(s.get('pass_at_1', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_grpo_loss = sum(s.get('grpo_loss', 0.0) for s in epoch_stats) / len(epoch_stats)
                total_syntax_errors = sum(s.get('syntax_errors', 0) for s in epoch_stats)
                
                logger.info(f"Epoch {epoch + 1} Summary:")
                logger.info(f"  Average Reward: {avg_reward:.4f}")
                logger.info(f"  Average Success Rate: {avg_success_rate:.4f}")
                logger.info(f"  Average Pass@1: {avg_pass_at_1:.4f}")
                logger.info(f"  Average Test Pass Rate: {avg_test_pass_rate:.4f}")
                logger.info(f"  Average GRPO Loss: {avg_grpo_loss:.4f}")
                logger.info(f"  Total Syntax Errors: {total_syntax_errors}")
                
                # Early stopping check (only enable early stopping when patience > 0)
                current_metric = avg_test_pass_rate
                
                if current_metric > self.best_test_pass_rate:
                    self.best_test_pass_rate = current_metric
                    self.best_reward = avg_reward
                    self.patience_counter = 0
                    
                    # Save best model
                    best_model_path = f"{self.config.training.output_dir}/best_epoch_model"
                    save_model_and_tokenizer(
                        self.model, 
                        self.tokenizer, 
                        best_model_path,
                        push_to_hub=False
                    )
                    logger.info(f"🎯 NEW BEST EPOCH! Test pass rate: {current_metric:.4f}")
                    
                elif self.patience > 0:  # Only check performance degradation when early stopping is enabled
                    self.patience_counter += 1
                    performance_drop = self.best_test_pass_rate - current_metric
                    logger.info(f"⚠️  Performance drop: {performance_drop:.4f} (patience: {self.patience_counter}/{self.patience})")
                    
                    if self.patience_counter >= self.patience:
                        logger.warning(f"🛑 EARLY STOPPING: No improvement for {self.patience} epochs")
                        self.early_stop = True
                
                # If early stopping is disabled, log but take no action
                if self.patience == 0 and current_metric <= self.best_test_pass_rate:
                    performance_drop = self.best_test_pass_rate - current_metric
                    logger.info(f"ℹ️  Performance drop: {performance_drop:.4f} (early stopping disabled)")
                
                # Log epoch summary to wandb
                if self.config.use_wandb:
                    wandb.log({
                        f"epoch_{epoch + 1}/avg_reward": avg_reward,
                        f"epoch_{epoch + 1}/avg_success_rate": avg_success_rate,
                        f"epoch_{epoch + 1}/avg_pass_at_1": avg_pass_at_1,
                        f"epoch_{epoch + 1}/avg_test_pass_rate": avg_test_pass_rate,
                        f"epoch_{epoch + 1}/avg_grpo_loss": avg_grpo_loss,
                        f"epoch_{epoch + 1}/syntax_errors": total_syntax_errors,
                        f"epoch_{epoch + 1}/patience_counter": self.patience_counter,
                        f"epoch_{epoch + 1}/best_test_pass_rate": self.best_test_pass_rate,
                    }, step=total_steps)
                
                # 💾 Save merged weights for each epoch (if enabled)
                if getattr(self.config, 'save_merged_epochs', False):
                    self._save_merged_epoch_model(epoch + 1, avg_test_pass_rate, avg_reward)
            
            # Check for early stopping
            if self.early_stop:
                logger.info(f"🏁 Training stopped early at epoch {epoch + 1}")
                break
        
        # Save final model
        logger.info("Training completed. Saving final model...")
        final_model_path = f"{self.config.training.output_dir}/final_model"
        save_model_and_tokenizer(
            self.model, 
            self.tokenizer, 
            final_model_path,
            push_to_hub=self.config.hf.push_to_hub,
            hub_model_id=self.config.hf.hub_model_id,
            hub_token=self.config.hf.hub_token
        )
        
        if self.config.use_wandb:
            wandb.finish()
        
        logger.info("Standard GRPO training completed successfully!")


def main():
    """Main function to run GRPO training."""
    # Load configuration
    from config import load_config_from_env
    config = load_config_from_env()
    config.method = "grpo"
    
    # Create trainer and run training
    trainer = CodeGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 