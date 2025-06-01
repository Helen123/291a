import torch
import torch.nn.functional as F
import logging
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from tqdm import tqdm
import random

from config import ExperimentConfig
from model_utils import load_tokenizer, load_base_model, setup_lora_model, save_model_and_tokenizer, calculate_model_size
from data_utils import create_mbpp_dataset_for_rl, RewardFunction, MBPPDataProcessor

logger = logging.getLogger(__name__)


class CodeDPOTrainer:
    """Direct Preference Optimization Trainer for code generation with Q-LoRA."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize DPO trainer.
        
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
        
        # DPO hyperparameters
        self.beta = 0.1  # DPO temperature parameter
        self.learning_rate = config.training.learning_rate
        self.num_candidates = 4  # Number of candidates to generate per prompt
        
        # Early stopping and best model tracking
        self.best_test_pass_rate = 0.0
        self.best_reward = 0.0
        self.patience = 2
        self.patience_counter = 0
        self.early_stop = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def setup_model(self):
        """Setup model and tokenizer for DPO training."""
        logger.info("Loading model and tokenizer for DPO training...")
        
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
        logger.info(f"DPO hyperparameters:")
        logger.info(f"  Beta (temperature): {self.beta}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Number of candidates per prompt: {self.num_candidates}")
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
        
    def generate_candidates(self, prompts: List[str]) -> List[List[str]]:
        """
        Generate multiple candidate responses for each prompt.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of lists of candidate responses
        """
        all_candidates = []
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
                
                candidates = []
                
                for _ in range(self.num_candidates):
                    # Generate response with some randomness
                    outputs = self.model.generate(
                        prompt_tokens['input_ids'],
                        attention_mask=prompt_tokens['attention_mask'],
                        max_new_tokens=self.config.training.max_new_tokens,
                        do_sample=True,  # Enable sampling for diversity
                        temperature=0.8,  # Higher temperature for diversity
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Extract generated sequence
                    generated_ids = outputs[0]
                    response_ids = generated_ids[prompt_tokens['input_ids'].shape[1]:]
                    
                    # Decode response
                    response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    candidates.append(response)
                
                all_candidates.append(candidates)
        
        return all_candidates
    
    def create_preference_pairs(self, prompts: List[str], candidates: List[List[str]], 
                               test_cases: List[str]) -> List[Dict]:
        """
        Create preference pairs based on test execution results.
        
        Args:
            prompts: List of input prompts
            candidates: List of lists of candidate responses
            test_cases: List of test cases
            
        Returns:
            List of preference pair dictionaries
        """
        preference_pairs = []
        
        for prompt, candidate_list, test_case in zip(prompts, candidates, test_cases):
            # Evaluate all candidates
            candidate_scores = []
            candidate_details = []
            
            for candidate in candidate_list:
                full_code = prompt + candidate
                
                # Get detailed evaluation
                rewards, details = self.reward_fn.compute_batch_rewards_detailed(
                    generated_codes=[full_code],
                    prompts=[prompt],
                    test_cases=[test_case],
                    max_workers=1
                )
                
                # Compute comprehensive score
                detail = details[0] if details else {}
                score = self.compute_preference_score(detail, rewards[0] if rewards else 0.0)
                
                candidate_scores.append(score)
                candidate_details.append(detail)
            
            # Find best and worst candidates
            if len(candidate_scores) >= 2:
                best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
                worst_idx = min(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
                
                # Only create pair if there's a meaningful difference
                if candidate_scores[best_idx] > candidate_scores[worst_idx] + 0.1:
                    preference_pairs.append({
                        'prompt': prompt,
                        'preferred': candidate_list[best_idx],
                        'rejected': candidate_list[worst_idx],
                        'preferred_score': candidate_scores[best_idx],
                        'rejected_score': candidate_scores[worst_idx],
                        'preferred_details': candidate_details[best_idx],
                        'rejected_details': candidate_details[worst_idx]
                    })
        
        return preference_pairs
    
    def compute_preference_score(self, details: Dict, base_reward: float) -> float:
        """
        Compute a comprehensive preference score for a candidate.
        
        Args:
            details: Detailed evaluation results
            base_reward: Base reward from evaluation
            
        Returns:
            Comprehensive preference score
        """
        score = base_reward
        
        # Bonus for passing all tests
        if details.get('test_ratio', 0.0) == 1.0:
            score += 0.5
        
        # Penalty for syntax errors
        if details.get('has_syntax_error', False):
            score -= 0.5
        
        # Small bonus for code quality
        score += details.get('quality_score', 0.0) * 0.1
        
        return score
    
    def compute_log_probs(self, model: torch.nn.Module, prompt: str, response: str) -> torch.Tensor:
        """
        Compute log probabilities for a response given a prompt.
        
        Args:
            model: Model to use for computation
            prompt: Input prompt
            response: Response to evaluate
            
        Returns:
            Log probabilities for the response tokens
        """
        device = next(model.parameters()).device
        
        # Combine prompt and response
        full_text = prompt + response
        
        # Tokenize
        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_seq_length
        ).to(device)
        
        # Get model outputs
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(**tokens)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probs for response tokens only
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        response_start = prompt_length
        response_end = tokens['input_ids'].shape[1]
        
        if response_end > response_start:
            response_log_probs = log_probs[0, response_start-1:response_end-1, :]
            response_tokens = tokens['input_ids'][0, response_start:response_end]
            
            # Select log probs for actual tokens
            selected_log_probs = response_log_probs.gather(
                1, response_tokens.unsqueeze(-1)
            ).squeeze(-1)
            
            return selected_log_probs.sum()
        else:
            return torch.tensor(0.0, device=device)
    
    def compute_dpo_loss(self, preference_pairs: List[Dict]) -> torch.Tensor:
        """
        Compute DPO loss for preference pairs.
        
        Args:
            preference_pairs: List of preference pair dictionaries
            
        Returns:
            DPO loss tensor
        """
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0
        
        for pair in preference_pairs:
            prompt = pair['prompt']
            preferred = pair['preferred']
            rejected = pair['rejected']
            
            # Compute log probabilities for preferred response
            policy_preferred_logp = self.compute_log_probs(self.model, prompt, preferred)
            ref_preferred_logp = self.compute_log_probs(self.ref_model, prompt, preferred)
            
            # Compute log probabilities for rejected response
            policy_rejected_logp = self.compute_log_probs(self.model, prompt, rejected)
            ref_rejected_logp = self.compute_log_probs(self.ref_model, prompt, rejected)
            
            # Compute log ratios
            preferred_ratio = policy_preferred_logp - ref_preferred_logp
            rejected_ratio = policy_rejected_logp - ref_rejected_logp
            
            # DPO loss: -log(sigma(beta * (preferred_ratio - rejected_ratio)))
            logits = self.beta * (preferred_ratio - rejected_ratio)
            loss = -F.logsigmoid(logits)
            
            total_loss += loss
            num_pairs += 1
        
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return total_loss
    
    def train_step(self, batch_data) -> Dict[str, float]:
        """
        Perform one DPO training step.
        
        Args:
            batch_data: Training batch
            
        Returns:
            Training statistics
        """
        prompts = batch_data["prompt"]
        test_cases = batch_data["test_cases"]
        
        # Generate multiple candidates for each prompt
        candidates = self.generate_candidates(prompts)
        
        # Create preference pairs based on test results
        preference_pairs = self.create_preference_pairs(prompts, candidates, test_cases)
        
        if not preference_pairs:
            # No valid preference pairs generated
            return {
                "dpo_loss": 0.0,
                "num_pairs": 0,
                "rewards/mean": 0.0,
                "success_rate": 0.0,
                "pass_at_1": 0.0,
                "test_pass_rate": 0.0,
            }
        
        # Compute DPO loss
        dpo_loss = self.compute_dpo_loss(preference_pairs)
        
        # Backward pass
        self.model.train()
        self.optimizer.zero_grad()
        dpo_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.max_grad_norm)
        
        self.optimizer.step()
        
        # Compute statistics
        preferred_scores = [pair['preferred_score'] for pair in preference_pairs]
        rejected_scores = [pair['rejected_score'] for pair in preference_pairs]
        all_scores = preferred_scores + rejected_scores
        
        # Compute detailed metrics
        all_details = []
        for pair in preference_pairs:
            all_details.append(pair['preferred_details'])
            all_details.append(pair['rejected_details'])
        
        valid_details = [d for d in all_details if d and not d.get('has_syntax_error', False)]
        
        if valid_details:
            total_passed = sum(d.get('passed_tests', 0) for d in all_details)
            total_tests = sum(d.get('total_tests', 0) for d in all_details)
            perfect_solutions = sum(1 for d in all_details if d.get('test_ratio', 0.0) == 1.0)
        else:
            total_passed = 0
            total_tests = 0
            perfect_solutions = 0
        
        syntax_errors = sum(1 for d in all_details if d.get('has_syntax_error', False))
        
        stats = {
            "dpo_loss": dpo_loss.item(),
            "num_pairs": len(preference_pairs),
            "rewards/mean": np.mean(all_scores) if all_scores else 0.0,
            "preferred_score_mean": np.mean(preferred_scores) if preferred_scores else 0.0,
            "rejected_score_mean": np.mean(rejected_scores) if rejected_scores else 0.0,
            "score_gap": np.mean(preferred_scores) - np.mean(rejected_scores) if preferred_scores and rejected_scores else 0.0,
            "success_rate": sum(1 for s in all_scores if s > 0) / len(all_scores) if all_scores else 0.0,
            "test_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            "pass_at_1": perfect_solutions / len(all_details) if all_details else 0.0,
            "syntax_errors": syntax_errors,
            "total_tests": total_tests,
            "passed_tests": total_passed,
        }
        
        return stats

    def train(self):
        """Run DPO training."""
        logger.info("Starting Direct Preference Optimization training with Q-LoRA...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_reward_function()
        
        # Initialize wandb if enabled
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name.replace("ppo", "dpo").replace("grpo", "dpo"),
                config=self.config.__dict__
            )
        
        # Training loop
        total_steps = 0
        best_success_rate = 0.0
        
        # Create batches from dataset
        batch_size = self.config.training.per_device_train_batch_size
        num_batches = len(self.dataset) // batch_size
        
        logger.info(f"Starting DPO training with {num_batches} batches per epoch")
        logger.info(f"Dataset size: {len(self.dataset)} samples")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Beta (temperature): {self.beta}")
        logger.info(f"Candidates per prompt: {self.num_candidates}")
        
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
                
                # Perform DPO training step
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
                        'DPOLoss': f"{step_stats.get('dpo_loss', 0.0):.3f}",
                        'Pairs': f"{step_stats.get('num_pairs', 0)}"
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
                avg_dpo_loss = sum(s.get('dpo_loss', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_score_gap = sum(s.get('score_gap', 0.0) for s in epoch_stats) / len(epoch_stats)
                total_pairs = sum(s.get('num_pairs', 0) for s in epoch_stats)
                total_syntax_errors = sum(s.get('syntax_errors', 0) for s in epoch_stats)
                
                logger.info(f"Epoch {epoch + 1} Summary:")
                logger.info(f"  Average Reward: {avg_reward:.4f}")
                logger.info(f"  Average Success Rate: {avg_success_rate:.4f}")
                logger.info(f"  Average Pass@1: {avg_pass_at_1:.4f}")
                logger.info(f"  Average Test Pass Rate: {avg_test_pass_rate:.4f}")
                logger.info(f"  Average DPO Loss: {avg_dpo_loss:.4f}")
                logger.info(f"  Average Score Gap: {avg_score_gap:.4f}")
                logger.info(f"  Total Preference Pairs: {total_pairs}")
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
                        f"epoch_{epoch + 1}/avg_dpo_loss": avg_dpo_loss,
                        f"epoch_{epoch + 1}/avg_score_gap": avg_score_gap,
                        f"epoch_{epoch + 1}/total_pairs": total_pairs,
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
        
        logger.info("Direct Preference Optimization training completed successfully!")


def main():
    """Main function to run DPO training."""
    # Load configuration
    from config import load_config_from_env
    config = load_config_from_env()
    config.method = "dpo"
    
    # Create trainer and run training
    trainer = CodeDPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 