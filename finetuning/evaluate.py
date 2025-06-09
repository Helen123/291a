#!/usr/bin/env python3
"""
Evaluation script for fine-tuned CodeLlama models.
Uses the EXACT same evaluation approach as bigcode-evaluation-harness.
"""

import os
import sys
import argparse
import logging
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import numpy as np

from data_utils import MBPPDataProcessor, RewardFunction, check_code_correctness, HumanEvalDataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BigCodeCompatibleEvaluator:
    """Evaluator that exactly matches bigcode-evaluation-harness behavior."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use for inference
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = MBPPDataProcessor()
        
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left"
        )
        
        # Fix attention mask issue by setting a proper pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        
        # Ensure pad_token is different from eos_token to avoid warnings
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # Use a different token for padding
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Resize token embeddings if we added new tokens
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model loaded successfully")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        logger.info(f"EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        
    def generate_code(self, prompt: str, max_new_tokens: int = 256, 
                     temperature: float = 0.1, num_samples: int = 1) -> List[str]:
        """
        Generate code for a given prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            num_samples: Number of samples to generate
            
        Returns:
            List of generated code strings
        """
        # Tokenize with proper attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=False,  # No padding needed for single input
            truncation=True
        )
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=num_samples,
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
        
        # Decode outputs
        generated_codes = []
        for output in outputs:
            # Decode full generation (including prompt)
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Apply bigcode-style postprocessing
            postprocessed = self.processor.postprocess_generation(generated_text, prompt)
            generated_codes.append(postprocessed)
        
        return generated_codes
    
    def evaluate_on_mbpp(self, split: str = "test", max_samples: Optional[int] = None,
                        max_new_tokens: int = 256, temperature: float = 0.1,
                        num_samples: int = 15) -> Dict[str, Any]:
        """
        Evaluate model on MBPP dataset using EXACT bigcode-evaluation-harness method.
        
        Args:
            split: Dataset split to evaluate on
            max_samples: Maximum number of samples to evaluate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_samples: Number of samples per problem for pass@k estimation
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on MBPP {split} split")
        
        # Load dataset exactly like bigcode
        dataset = load_dataset("mbpp")[split]
        
        # Validate dataset size for test split (like bigcode)
        if split == "test":
            assert len(dataset) == 500, \
                "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Evaluating on {len(dataset)} problems")
        
        # Prepare evaluation data exactly like bigcode
        generations = []
        references = []
        
        for i, example in enumerate(tqdm(dataset, desc="Generating code")):
            # Create prompt exactly like bigcode MBPP task
            prompt = self.processor.create_prompt(example)
            
            # Generate code
            try:
                generated_codes = self.generate_code(
                    prompt, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_samples=num_samples
                )
                
                generations.append(generated_codes)
                
                # Reference is all test cases joined (like bigcode)
                reference = "\n".join(example["test_list"])
                references.append(reference)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i+1}/{len(dataset)} problems")
                    
            except Exception as e:
                logger.error(f"Error generating for task {example['task_id']}: {e}")
                # Add empty generation to maintain alignment
                generations.append([""] * num_samples)
                references.append("\n".join(example["test_list"]))
                continue
        
        # Evaluate using bigcode's compute_code_eval equivalent
        logger.info("Running code evaluation...")
        results = self.compute_code_eval_bigcode_style(generations, references)
        
        # Add additional statistics
        total_problems = len(generations)
        
        # Count problems with at least one passing solution
        problems_with_pass = 0
        for gen_list in generations:
            has_pass = False
            for i, generation in enumerate(gen_list):
                if i < len(references):
                    ref = references[generations.index(gen_list)] if gen_list in generations else ""
                    if ref:
                        # Extract code from postprocessed generation
                        prompt = self.processor.create_prompt(dataset[generations.index(gen_list)])
                        if generation.startswith(prompt):
                            code = generation[len(prompt):]
                        else:
                            code = generation
                        
                        # Test code
                        test_result = check_code_correctness(code, ref, timeout=10.0)
                        if test_result["passed"]:
                            has_pass = True
                            break
            
            if has_pass:
                problems_with_pass += 1
        
        results.update({
            "total_problems": total_problems,
            "problems_with_solutions": problems_with_pass,
            "solve_rate": problems_with_pass / total_problems if total_problems > 0 else 0,
            "generations": generations,
            "references": references
        })
        
        return results
    
    def compute_code_eval_bigcode_style(self, predictions: List[List[str]], 
                                      references: List[str]) -> Dict[str, float]:
        """
        Compute pass@k metrics exactly like bigcode-evaluation-harness.
        
        Args:
            predictions: List of lists containing generated code
            references: List of reference test cases
            
        Returns:
            Dictionary with pass@k metrics
        """
        def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
            """Estimate pass@k using unbiased estimator from HumanEval paper."""
            if num_samples - num_correct < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))
        
        # Evaluate each problem
        problem_results = []
        
        for i, (pred_list, reference) in enumerate(zip(predictions, references)):
            # Count correct solutions for this problem
            correct_count = 0
            
            for prediction in pred_list:
                try:
                    # Extract code from prediction (remove prompt if present)
                    # Get the corresponding prompt for this problem
                    dataset = load_dataset("mbpp")["test"]
                    if i < len(dataset):
                        prompt = self.processor.create_prompt(dataset[i])
                        if prediction.startswith(prompt):
                            code = prediction[len(prompt):]
                        else:
                            code = prediction
                    else:
                        code = prediction
                    
                    # Test the code
                    result = check_code_correctness(code, reference, timeout=10.0)
                    if result["passed"]:
                        correct_count += 1
                        
                except Exception:
                    # If execution fails, count as incorrect
                    continue
            
            problem_results.append({
                "total": len(pred_list),
                "correct": correct_count
            })
        
        # Compute pass@k for different k values
        k_values = [1, 5, 10, 15]
        results = {}
        
        for k in k_values:
            if all(r["total"] >= k for r in problem_results):
                pass_at_k_scores = []
                for result in problem_results:
                    score = estimate_pass_at_k(result["total"], result["correct"], k)
                    pass_at_k_scores.append(score)
                
                results[f"pass@{k}"] = np.mean(pass_at_k_scores)
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        # Remove large data structures for JSON serialization
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ["generations", "references"]}
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    def evaluate_on_humaneval(self, split: str = "test", max_samples: Optional[int] = None,
                            max_new_tokens: int = 256, temperature: float = 0.1,
                            num_samples: int = 15) -> Dict[str, Any]:
        """
        Evaluate model on HumanEval dataset using bigcode-compatible evaluation.
        
        Args:
            split: Dataset split (only "test" available for HumanEval)
            max_samples: Maximum number of problems to evaluate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_samples: Number of samples per problem
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Loading HumanEval {split} dataset...")
        processor = HumanEvalDataProcessor(split=split)
        dataset = processor.load_dataset()
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Evaluating on {len(dataset)} problems with {num_samples} samples each...")
        
        predictions = []
        references = []
        problems_with_pass = 0
        
        for i, problem in enumerate(tqdm(dataset, desc="Evaluating problems")):
            # Get prompt and reference
            prompt = processor.get_prompt(problem)
            reference = processor.get_reference(problem)
            
            # Generate multiple solutions
            generated_codes = self.generate_code(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_samples=num_samples
            )
            
            # Postprocess generations (bigcode-style)
            postprocessed_codes = []
            for code in generated_codes:
                postprocessed = processor.postprocess_generation(code, i)
                postprocessed_codes.append(postprocessed)
            
            predictions.append(postprocessed_codes)
            references.append(reference)
            
            # Check if any solution passes
            has_pass = False
            for code in postprocessed_codes:
                try:
                    # Extract the generated part (remove prompt)
                    if code.startswith(prompt):
                        code_only = code[len(prompt):]
                    else:
                        code_only = code
                    
                    # Test execution
                    result = check_code_correctness(code_only + reference, "", timeout=10.0)
                    if result["passed"]:
                        has_pass = True
                        break
                except Exception:
                    continue
            
            if has_pass:
                problems_with_pass += 1
        
        # Compute pass@k metrics using bigcode method
        results = self.compute_code_eval_humaneval_style(predictions, references)
        
        total_problems = len(dataset)
        results.update({
            "total_problems": total_problems,
            "problems_with_solutions": problems_with_pass,
            "solve_rate": problems_with_pass / total_problems if total_problems > 0 else 0,
            "generations": predictions,
            "references": references
        })
        
        return results

    def compute_code_eval_humaneval_style(self, predictions: List[List[str]], 
                                        references: List[str]) -> Dict[str, float]:
        """
        Compute pass@k metrics for HumanEval exactly like bigcode-evaluation-harness.
        
        Args:
            predictions: List of lists containing generated code (postprocessed)
            references: List of reference test cases with entry point
            
        Returns:
            Dictionary with pass@k metrics
        """
        def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
            """Estimate pass@k using unbiased estimator from HumanEval paper."""
            if num_samples - num_correct < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))
        
        # Evaluate each problem
        problem_results = []
        
        for i, (pred_list, reference) in enumerate(zip(predictions, references)):
            # Count correct solutions for this problem
            correct_count = 0
            
            for prediction in pred_list:
                try:
                    # For HumanEval, the prediction should already be postprocessed
                    # and include the prompt. We need to extract just the generated part
                    # and combine it with the test reference
                    
                    # Get the original prompt for this problem
                    processor = HumanEvalDataProcessor()
                    dataset = processor.load_dataset()
                    prompt = processor.get_prompt(dataset[i])
                    
                    # Extract generated code
                    if prediction.startswith(prompt):
                        code = prediction[len(prompt):]
                    else:
                        code = prediction
                    
                    # Combine code with test reference
                    test_program = code + reference
                    
                    # Test the code using timeout execution
                    result = check_code_correctness(test_program, "", timeout=3.0)
                    if result["passed"]:
                        correct_count += 1
                        
                except Exception:
                    # If execution fails, count as incorrect
                    continue
            
            problem_results.append({
                "total": len(pred_list),
                "correct": correct_count
            })
        
        # Compute pass@k for different k values (HumanEval standard)
        k_values = [1, 10, 100]
        results = {}
        
        for k in k_values:
            if all(r["total"] >= k for r in problem_results):
                pass_at_k_scores = []
                for result in problem_results:
                    score = estimate_pass_at_k(result["total"], result["correct"], k)
                    pass_at_k_scores.append(score)
                
                results[f"pass@{k}"] = np.mean(pass_at_k_scores)
        
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CodeLlama model using bigcode-compatible method")
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the fine-tuned model"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mbpp",
        choices=["mbpp", "humaneval"],
        help="Dataset to evaluate on (default: mbpp)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "validation"],
        help="Dataset split to evaluate on (default: test)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
        help="Number of samples per problem for pass@k estimation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for inference"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up environment for code execution
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    
    # Initialize evaluator
    evaluator = BigCodeCompatibleEvaluator(args.model_path, args.device)
    evaluator.load_model()
    
    # Run evaluation
    logger.info("Starting bigcode-compatible evaluation...")
    
    if args.dataset == "mbpp":
        results = evaluator.evaluate_on_mbpp(
            split=args.split,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_samples=args.num_samples
        )
    elif args.dataset == "humaneval":
        results = evaluator.evaluate_on_humaneval(
            split="test",  # HumanEval only has test split
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_samples=args.num_samples
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Print results
    logger.info("=" * 80)
    logger.info("BIGCODE-COMPATIBLE EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset.upper()} {args.split if args.dataset == 'mbpp' else 'test'}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Total problems: {results['total_problems']}")
    logger.info(f"Problems with solutions: {results['problems_with_solutions']}")
    logger.info(f"Solve rate: {results['solve_rate']:.3f}")
    
    for metric, value in results.items():
        if metric.startswith("pass@"):
            logger.info(f"{metric}: {value:.3f}")
    
    logger.info("=" * 80)
    
    # Save results
    if args.output_path is None:
        model_name = Path(args.model_path).name
        split_name = args.split if args.dataset == "mbpp" else "test"
        args.output_path = f"bigcode_eval_results_{args.dataset}_{model_name}_{split_name}.json"
    
    evaluator.save_results(results, args.output_path)
    
    logger.info("Bigcode-compatible evaluation completed successfully!")


if __name__ == "__main__":
    main() 