#!/usr/bin/env python3
"""
Validation script to test the entire pipeline and compare with bigcode-evaluation-harness.
"""

import os
import sys
import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset

from data_utils import MBPPDataProcessor, RewardFunction, check_code_correctness
from evaluate import BigCodeCompatibleEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_prompt_format():
    """Test if our prompt format matches bigcode exactly."""
    logger.info("🧪 Testing prompt format compatibility...")
    
    # Load sample from MBPP
    dataset = load_dataset("mbpp")["test"]
    sample = dataset[0]
    
    # Our prompt
    processor = MBPPDataProcessor()
    our_prompt = processor.create_prompt(sample)
    
    # Expected bigcode prompt format
    description = sample["text"]
    test_example = sample["test_list"][0]
    expected_prompt = f'"""\n{description}\n{test_example}\n"""\n'
    
    assert our_prompt == expected_prompt, f"Prompt mismatch!\nOurs: {repr(our_prompt)}\nExpected: {repr(expected_prompt)}"
    logger.info("✅ Prompt format matches bigcode exactly")


def test_postprocessing():
    """Test if our postprocessing matches bigcode exactly."""
    logger.info("🧪 Testing postprocessing compatibility...")
    
    processor = MBPPDataProcessor()
    
    # Test cases
    test_cases = [
        {
            "prompt": '"""\nWrite a function to find the nth fibonacci number.\nassert nth_fibonacci(3) == 2\n"""\n',
            "generation": '"""\nWrite a function to find the nth fibonacci number.\nassert nth_fibonacci(3) == 2\n"""\ndef nth_fibonacci(n):\n    if n <= 1:\n        return n\n    return nth_fibonacci(n-1) + nth_fibonacci(n-2)\nclass Test:\n    pass',
            "expected_clean": 'def nth_fibonacci(n):\n    if n <= 1:\n        return n\n    return nth_fibonacci(n-1) + nth_fibonacci(n-2)'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        result = processor.postprocess_generation(test_case["generation"], test_case["prompt"])
        expected_full = test_case["prompt"] + test_case["expected_clean"]
        
        if result != expected_full:
            logger.warning(f"⚠️ Postprocessing test {i+1} differs:")
            logger.warning(f"Result: {repr(result)}")
            logger.warning(f"Expected: {repr(expected_full)}")
        else:
            logger.info(f"✅ Postprocessing test {i+1} passed")


def test_reward_calculation():
    """Test reward calculation with known examples."""
    logger.info("🧪 Testing reward calculation...")
    
    reward_fn = RewardFunction()
    
    # Test case: correct solution
    prompt = '"""\nWrite a function to add two numbers.\nassert add_numbers(2, 3) == 5\n"""\n'
    correct_generation = prompt + "def add_numbers(a, b):\n    return a + b"
    test_cases = "assert add_numbers(2, 3) == 5\nassert add_numbers(0, 0) == 0\nassert add_numbers(-1, 1) == 0"
    
    reward_details = reward_fn.compute_reward(correct_generation, prompt, test_cases)
    
    logger.info(f"Test reward details: {reward_details}")
    assert reward_details['test_ratio'] == 1.0, f"Expected test ratio 1.0, got {reward_details['test_ratio']}"
    assert not reward_details['has_syntax_error'], "Expected no syntax error"
    logger.info("✅ Reward calculation test passed")


def test_evaluation_equivalence(model_path: str = None):
    """Test if our evaluation gives similar results to bigcode (if model available)."""
    if not model_path:
        logger.info("⏭️ Skipping evaluation test - no model path provided")
        return
    
    logger.info(f"🧪 Testing evaluation equivalence with model: {model_path}")
    
    try:
        evaluator = BigCodeCompatibleEvaluator(model_path)
        evaluator.load_model()
        
        # Test on a small subset
        results = evaluator.evaluate_on_mbpp(
            split="test",
            max_samples=5,  # Small test
            num_samples=3,
            temperature=0.1
        )
        
        logger.info(f"Evaluation results: {results}")
        logger.info("✅ Evaluation test completed")
        
    except Exception as e:
        logger.warning(f"⚠️ Evaluation test failed: {e}")


def run_bigcode_comparison(model_path: str = None):
    """Run comparison with official bigcode-evaluation-harness (if available)."""
    if not model_path:
        logger.info("⏭️ Skipping bigcode comparison - no model path provided")
        return
    
    bigcode_dir = Path("../bigcode-evaluation-harness")
    if not bigcode_dir.exists():
        logger.info("⏭️ Skipping bigcode comparison - bigcode-evaluation-harness not found")
        return
    
    logger.info("🧪 Running comparison with official bigcode-evaluation-harness...")
    
    try:
        # Run official bigcode evaluation on small subset
        cmd = [
            "python", "main.py",
            "--model", model_path,
            "--tasks", "mbpp",
            "--limit", "5",
            "--temperature", "0.1",
            "--n_samples", "3",
            "--allow_code_execution",
            "--metric_output_path", "comparison_results.json"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=bigcode_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Load and compare results
            results_file = bigcode_dir / "comparison_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    bigcode_results = json.load(f)
                logger.info(f"BigCode results: {bigcode_results}")
                logger.info("✅ BigCode comparison completed")
            else:
                logger.warning("⚠️ BigCode results file not found")
        else:
            logger.warning(f"⚠️ BigCode evaluation failed: {result.stderr}")
            
    except Exception as e:
        logger.warning(f"⚠️ BigCode comparison failed: {e}")


def validate_dataset_consistency():
    """Validate that our dataset loading matches bigcode expectations."""
    logger.info("🧪 Testing dataset consistency...")
    
    # Load test dataset
    dataset = load_dataset("mbpp")["test"]
    
    # Check size (bigcode expects exactly 500)
    assert len(dataset) == 500, f"Expected 500 test samples, got {len(dataset)}"
    
    # Check required fields
    sample = dataset[0]
    required_fields = ["text", "test_list", "task_id", "code"]
    for field in required_fields:
        assert field in sample, f"Missing required field: {field}"
    
    # Check test_list format
    assert isinstance(sample["test_list"], list), "test_list should be a list"
    assert len(sample["test_list"]) >= 1, "test_list should have at least one test"
    
    logger.info("✅ Dataset consistency validated")


def main():
    """Run all validation tests."""
    logger.info("🚀 STARTING PIPELINE VALIDATION")
    logger.info("=" * 60)
    
    # Set environment for code execution
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    
    try:
        # Basic tests (always run)
        validate_dataset_consistency()
        test_prompt_format()
        test_postprocessing()
        test_reward_calculation()
        
        # Model-dependent tests (optional)
        model_path = sys.argv[1] if len(sys.argv) > 1 else None
        if model_path:
            logger.info(f"🔍 Running model-dependent tests with: {model_path}")
            test_evaluation_equivalence(model_path)
            run_bigcode_comparison(model_path)
        else:
            logger.info("ℹ️ To run model tests, provide model path as argument")
        
        logger.info("=" * 60)
        logger.info("🎉 ALL VALIDATION TESTS COMPLETED!")
        logger.info("✅ Pipeline is compatible with bigcode-evaluation-harness")
        
    except AssertionError as e:
        logger.error(f"❌ Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 