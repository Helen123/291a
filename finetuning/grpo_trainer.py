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
from data_utils import create_mbpp_dataset_for_rl, RewardFunction, MBPPDataProcessor

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
        
        # GRPO hyperparameters
        self.alpha = config.training.grpo_alpha  # GRPO scaling factor
        self.learning_rate = config.training.learning_rate
        self.num_samples = 4  # Number of samples per prompt for GRPO
        
        # Early stopping and best model tracking
        self.best_test_pass_rate = 0.0
        self.best_reward = 0.0
        self.patience = 2
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
        logger.info(f"  Alpha (scaling factor): {self.alpha}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Number of samples per prompt: {self.num_samples}")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        
    def setup_data(self):
        """Setup training dataset."""
        logger.info("Loading and preprocessing MBPP dataset...")
        
        # Load dataset
        self.dataset = create_mbpp_dataset_for_rl(
            split="train",
            max_samples=None  # Use full dataset
        )
        
        # Filter out examples that are too long
        def filter_long_examples(example):
            prompt_length = len(self.tokenizer.encode(example["prompt"]))
            return prompt_length <= self.config.data.max_prompt_length
        
        self.dataset = self.dataset.filter(filter_long_examples)
        
        logger.info(f"Dataset loaded with {len(self.dataset)} examples")
        
    def setup_reward_function(self):
        """Setup reward function for code execution."""
        logger.info("Setting up reward function...")
        
        self.reward_fn = RewardFunction(
            timeout=self.config.training.code_execution_timeout,
            max_reward=1.0,
            min_reward=0.0,
            syntax_penalty=-0.1
        )
        
    def generate_responses(self, prompts: List[str]) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
        """
        Generate multiple responses for each prompt.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            Tuple of (responses, log_probs) for each prompt
        """
        all_responses = []
        all_log_probs = []
        
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
                    # Generate response
                    outputs = self.model.generate(
                        prompt_tokens['input_ids'],
                        attention_mask=prompt_tokens['attention_mask'],
                        max_new_tokens=self.config.training.max_new_tokens,
                        do_sample=self.config.training.do_sample,
                        temperature=self.config.training.temperature,
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
        Compute standard GRPO loss.
        
        Args:
            log_probs: Log probabilities from policy model
            ref_log_probs: Log probabilities from reference model  
            rewards: Reward values for each sample
            
        Returns:
            GRPO loss tensor
        """
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        num_samples = 0
        
        for prompt_log_probs, prompt_ref_log_probs, prompt_rewards in zip(log_probs, ref_log_probs, rewards):
            if len(prompt_rewards) < 2:
                continue  # Need at least 2 samples for comparison
            
            # Convert rewards to advantages (zero-mean)
            rewards_tensor = torch.tensor(prompt_rewards, device=device, dtype=torch.float32)
            advantages = rewards_tensor - rewards_tensor.mean()
            
            # Compute log probability ratios
            for i, (lp, ref_lp, adv) in enumerate(zip(prompt_log_probs, prompt_ref_log_probs, advantages)):
                if len(lp) == 0 or len(ref_lp) == 0:
                    continue
                
                # Ensure tensors have same length
                min_len = min(len(lp), len(ref_lp))
                if min_len == 0:
                    continue
                    
                lp = lp[:min_len]
                ref_lp = ref_lp[:min_len]
                
                # Compute log ratio (policy - reference)
                log_ratio = (lp - ref_lp).sum()
                
                # GRPO loss: -alpha * advantage * log_ratio
                loss = -self.alpha * adv * log_ratio
                total_loss += loss
                num_samples += 1
        
        if num_samples > 0:
            total_loss = total_loss / num_samples
        
        return total_loss
    
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
        
        # Generate responses using current policy
        responses, log_probs = self.generate_responses(prompts)
        
        # Compute reference log probabilities
        ref_log_probs = self.compute_reference_log_probs(prompts, responses)
        
        # Compute rewards
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
        
        # Compute GRPO loss
        grpo_loss = self.compute_grpo_loss(log_probs, ref_log_probs, all_rewards)
        
        # Backward pass
        self.model.train()
        self.optimizer.zero_grad()
        grpo_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.max_grad_norm)
        
        self.optimizer.step()
        
        # Compute statistics
        flat_rewards = [r for reward_list in all_rewards for r in reward_list]
        flat_responses = [r for response_list in responses for r in response_list]
        
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
        logger.info(f"Alpha (scaling factor): {self.alpha}")
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
                
                # Early stopping check
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
                    
                else:
                    self.patience_counter += 1
                    performance_drop = self.best_test_pass_rate - current_metric
                    logger.info(f"⚠️  Performance drop: {performance_drop:.4f} (patience: {self.patience_counter}/{self.patience})")
                    
                    if self.patience_counter >= self.patience:
                        logger.warning(f"🛑 EARLY STOPPING: No improvement for {self.patience} epochs")
                        self.early_stop = True
                
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