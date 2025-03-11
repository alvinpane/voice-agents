#!/usr/bin/env python3
import json
import os
import subprocess
import argparse
import time
import glob
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# IVR tree structure
IVR_TREE = {
  "Main Menu": {
    # "1. Account Information": {
    #   "1.1. Check Balance": {
    #     "1.1.1. Current Balance": {},
    #     "1.1.2. Available Credit": {},
    #     "1.1.3. Last Statement Balance": {}
    #   },
    #   "1.2. Recent Transactions": {
    #     "1.2.1. Last 5 Transactions": {},
    #     "1.2.2. Transactions Over $500": {},
    #     "1.2.3. Specific Date Range": {}
    #   },
    #   "1.3. Payment Due Date": {
        # "1.3.1. Current Payment Due Date": {},
        # "1.3.2. Minimum Payment": {},
        # "1.3.3. Payment History": {}
    #   }
    # },
    # "2. Rewards": {
    #   "2.1. Rewards Balance": {
    #     "2.1.1. Points Balance": {},
    #     "2.1.2. Cashback Balance": {},
    #     "2.1.3. Expiring Rewards": {}
    #   },
    #   "2.2. Redeem Rewards": {
    #     "2.2.1. Redeem for Travel": {},
    #     "2.2.2. Redeem for Gift Cards": {},
    #     "2.2.3. Redeem for Statement Credit": {}
    #   },
    #   "2.3. Explore Offers": {
    #     "2.3.1. Travel Offers": {},
    #     "2.3.2. Dining Offers": {},
    #     "2.3.3. Shopping Offers": {}
    #   }
    # },
    "3. Lost or Stolen Card": {
    #   "3.1. Report Lost Card": {
    #     "3.1.1. Block Current Card": {},
    #     "3.1.2. Issue Replacement Card": {},
    #     "3.1.3. Check Card Status": {}
    #   },
      "3.2. Request Replacement": {
        # "3.2.1. Standard Delivery": {},
        "3.2.2. Expedited Delivery": {},
        "3.2.3. Digital Card Activation": {}
      },
      "3.3. Freeze Account": {
        "3.3.1. Temporary Freeze": {},
        "3.3.2. Permanent Freeze": {},
        "3.3.3. Freeze History": {}
      }
    },
    "4. Customer Support": {
      "4.1. Speak to an Agent": {
        "4.1.1. Billing Queries": {},
        "4.1.2. Technical Assistance": {},
        "4.1.3. General Inquiries": {}
      },
      "4.2. FAQs": {
        "4.2.1. Billing FAQs": {},
        "4.2.2. Rewards FAQs": {},
        "4.2.3. Security FAQs": {}
      },
      "4.3. Feedback Submission": {
        "4.3.1. Rate Service": {},
        "4.3.2. Submit Complaint": {},
        "4.3.3. Share Suggestions": {}
      }
    },
    "5. Technical Support": {
      "5.1. Mobile App Assistance": {
        "5.1.1. App Installation Issues": {},
        "5.1.2. Login Problems": {},
        "5.1.3. Payment Processing Errors": {}
      },
      "5.2. Online Banking Issues": {
        "5.2.1. Forgotten Password": {},
        "5.2.2. Account Lockout": {},
        "5.2.3. Transaction Failures": {}
      },
      "5.3. Card Activation Problems": {
        "5.3.1. Activation Code Issues": {},
        "5.3.2. Invalid Card Details": {},
        "5.3.3. Activation Delays": {}
      }
    }
  }
}

def extract_leaf_nodes(tree, prefix=None, result=None):
    """
    Extract all leaf nodes (items with 2 decimal places in their index) from the IVR tree.
    Returns a list of tuples with (index, task_name).
    """
    if result is None:
        result = []
    
    if prefix is None:
        # Start with the main menu
        for main_key, main_value in tree.items():
            extract_leaf_nodes(main_value, "", result)
    else:
        for key, value in tree.items():
            # Extract the index part (e.g., "1.1.1" from "1.1.1. Current Balance")
            index_parts = key.split(". ", 1)
            if len(index_parts) < 2:
                continue
                
            index = index_parts[0]
            name = index_parts[1]
            
            # Check if this is a leaf node (has two dots in the index)
            if index.count('.') == 2:
                # Convert index to hyphenated format (e.g., "1-1-1")
                hyphenated_index = index.replace('.', '-')
                task_name = name.lower().replace(' ', '_')
                result.append((hyphenated_index, task_name))
            elif value:  # If not a leaf but has children
                extract_leaf_nodes(value, index, result)
                
    return result

def run_test(test_id, test_task, max_duration=180):
    """
    Run a single test using multi-bot.py with the specified parameters.
    
    Returns:
        Tuple of (success, result_file_path) where success is a boolean and
        result_file_path is the path to the generated result file
    """
    print(f"Running test for {test_task} (ID: {test_id})...")
    
    # Build the command
    cmd = [
        "python", "multi-bot.py",
        "--test_task", test_task,
        "--test_id", test_id,
        "--max_duration", str(max_duration)
    ]
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Test completed successfully for {test_task}")
        
        # Find the result file that was just created
        result_file_pattern = f"results/test_{test_task}_{test_id}.json"
        matching_files = glob.glob(result_file_pattern)
        
        if matching_files:
            return True, matching_files[0]
        else:
            print(f"Warning: Test executed but couldn't find result file matching {result_file_pattern}")
            return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error running test for {test_task}: {e}")
        return False, None

def generate_test_summary(tests_run, tests_failed, llm_evaluations=None):
    """
    Generate a summary report of all tests, including LLM evaluations if available.
    
    Args:
        tests_run: List of test details that were run
        tests_failed: List of tests that failed during execution
        llm_evaluations: Dictionary of LLM evaluation results {test_id: {"passed": bool, "reasoning": str}}
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_file = f"results/test_summary_{now}.json"
    
    # Calculate success metrics
    execution_success = len(tests_run) - len(tests_failed)
    
    # If we have LLM evaluations, calculate those metrics too
    llm_success_count = 0
    if llm_evaluations:
        for evaluation in llm_evaluations.values():
            if evaluation.get("passed", False):
                llm_success_count += 1
    
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(tests_run),
        "execution": {
            "successful_tests": execution_success,
            "failed_tests": len(tests_failed),
            "failure_details": tests_failed
        },
        "tests_run": tests_run
    }
    
    # Add LLM evaluation results if available
    if llm_evaluations:
        summary["llm_evaluation"] = {
            "successful_tests": llm_success_count,
            "failed_tests": len(llm_evaluations) - llm_success_count,
            "pass_rate": f"{(llm_success_count / len(llm_evaluations) * 100):.2f}%" if llm_evaluations else "N/A",
            "details": llm_evaluations
        }
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Write summary to file
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nTest summary written to {summary_file}")
    
    # Also print summary to console
    print("\n=== Test Summary ===")
    print(f"Total tests: {len(tests_run)}")
    print(f"Execution successful: {execution_success}")
    print(f"Execution failed: {len(tests_failed)}")
    
    if llm_evaluations:
        print(f"\nLLM Evaluation Results:")
        print(f"Passed: {llm_success_count}/{len(llm_evaluations)} ({(llm_success_count / len(llm_evaluations) * 100):.2f}%)")
        
        # Print failed evaluations with reasons
        failed_evaluations = {k: v for k, v in llm_evaluations.items() if not v.get("passed", False)}
        if failed_evaluations:
            print("\nFailed LLM evaluations:")
            for test_id, eval_data in failed_evaluations.items():
                task = next((test["task"] for test in tests_run if test["id"] == test_id), "unknown")

def evaluate_test_with_llm(test_id, test_task, result_file):
    """
    Use OpenAI's LLM to evaluate if the test was successful based on the transcript.
    
    Args:
        test_id: The ID of the test (e.g., "1-1-1")
        test_task: The task being tested (e.g., "current_balance")
        result_file: Path to the JSON result file containing the transcript
        
    Returns:
        A tuple of (success, feedback) where success is a boolean and feedback is a string
    """
    # Load environment variables for API keys
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(result_file)
    # Load the test result file
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
    except Exception as e:
        return False, f"Failed to load result file: {e}"
    
    # Extract the transcripts
    transcripts = result_data.get("transcripts", {})
    if not transcripts:
        return False, "No transcripts found in result file"
    
    # Combine all transcripts into a single text for analysis
    combined_transcript = ""
    for bot_name, transcript in transcripts.items():
        combined_transcript += f"==== {bot_name} Transcript ====\n"
        combined_transcript += transcript + "\n\n"
    
    # Create the prompt for the LLM
    prompt = f"""
You are evaluating an IVR (Interactive Voice Response) test for an American Express customer service system.

Test ID: {test_id}
Test Task: {test_task}

The objective of this test was for a customer to successfully inquire about their "{test_task}".

Here is the transcript of the conversation between the customer and the AI agent:

{combined_transcript}

Based on this transcript, please determine if the test was successful by checking if ALL of the following criteria were met:
1. The customer's inquiry about {test_task} was clearly addressed
2. The agent provided specific information related to {test_task}
3. The conversation reached a natural conclusion
4. The conversation stayed focused on the test task

Please provide your evaluation in the following format:
PASS: [true/false]
"""

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change the model as needed
            messages=[
                {"role": "system", "content": "You are an objective evaluator of customer service transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for more consistent evaluations
        )
        
        # Extract the response text
        evaluation = response.choices[0].message.content
        print(evaluation)
        # Parse the evaluation to determine if the test passed
        if 'true' in evaluation.lower():
            passed=True
        else:    
            passed = False
        
        return passed
    
    except Exception as e:
        return False, f"Error calling OpenAI API: {e}"

def main():
    """
    Main function to parse arguments and run tests.
    """
    parser = argparse.ArgumentParser(description="Run automated tests for IVR system")
    parser.add_argument("--max_duration", type=int, default=180,
                       help="Maximum duration for each test in seconds (default: 180)")
    parser.add_argument("--specific_test", type=str, default=None,
                       help="Run a specific test by providing the hyphenated index (e.g., '1-1-1')")
    parser.add_argument("--delay", type=int, default=5,
                       help="Delay between tests in seconds (default: 5)")
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only evaluate existing test results without running new tests")
    args = parser.parse_args()
    
    # Extract all leaf nodes from the IVR tree
    leaf_nodes = extract_leaf_nodes(IVR_TREE)
    
    # Filter for a specific test if requested
    if args.specific_test:
        leaf_nodes = [node for node in leaf_nodes if node[0] == args.specific_test]
        if not leaf_nodes:
            print(f"No test found with ID {args.specific_test}")
            return
    
    print(f"Found {len(leaf_nodes)} tests to run")
    
    # Initialize collections
    tests_run = []
    tests_failed = []
    result_files = {}  # Map of test_id to result file paths
    llm_evaluations = {}  # Will store LLM evaluation results
    
    if not args.evaluate_only:
        # Run the tests
        for i, (test_id, test_task) in enumerate(leaf_nodes):
            print(f"\nTest {i+1} of {len(leaf_nodes)}: {test_id} - {test_task}")
            
            tests_run.append({
                "id": test_id,
                "task": test_task,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            success, result_file = run_test(test_id, test_task, args.max_duration)
            
            if not success:
                tests_failed.append({
                    "id": test_id,
                    "task": test_task
                })
            elif result_file:  # If the test ran successfully and we have a result file
                result_files[test_id] = result_file
                
                # Evaluate the test immediately after it runs
                print(f"Evaluating test {test_id} - {test_task} with LLM...")
                passed = evaluate_test_with_llm(test_id, test_task, result_file)
                
                llm_evaluations[test_id] = {
                    "passed": passed,
                }
                
                # Print the evaluation result
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status} - {test_id} ({test_task})")
            
            # Delay between tests (except after the last test)
            if i < len(leaf_nodes) - 1:
                print(f"Waiting {args.delay} seconds before next test...")
                time.sleep(args.delay)
    else:
        # Evaluate only mode - find existing result files
        print("Evaluation-only mode - finding existing test results...")
        
        for test_id, test_task in leaf_nodes:
            # Find result files for this test
            result_file_pattern = f"results/test_{test_task}_{test_id}.json"
            matching_files = glob.glob(result_file_pattern)
            
            if matching_files:
                newest_file = max(matching_files, key=os.path.getctime)
                result_files[test_id] = newest_file
                tests_run.append({
                    "id": test_id,
                    "task": test_task,
                    "timestamp": datetime.fromtimestamp(os.path.getctime(newest_file)).strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"Found existing result for {test_id} - {test_task}: {newest_file}")
            else:
                print(f"No existing results found for {test_id} - {test_task}")
    
    # Evaluate tests with LLM
    print("\n=== Evaluating Test Results with LLM ===")
    for test_id, result_file in result_files.items():
        if not result_file:
            continue
            
        test_task = next((test["task"] for test in tests_run if test["id"] == test_id), None)
        if not test_task:
            continue
            
        print(f"Evaluating test {test_id} - {test_task}...")
        passed = evaluate_test_with_llm(test_id, test_task, result_file)
        
        llm_evaluations[test_id] = {
            "passed": passed        }
        
        # Print the result
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_id} ({test_task})")
    
    # Generate test summary with LLM evaluations
    generate_test_summary(tests_run, tests_failed, llm_evaluations)

def find_existing_result_files():
    """
    Find and categorize all existing test result files.
    Returns a dictionary of {test_id: {"file": file_path, "task": task_name}}
    """
    results = {}
    result_files = glob.glob("results/test_*.json")
    
    for file_path in result_files:
        # Skip summary files
        if "summary" in file_path:
            continue
            
        filename = os.path.basename(file_path)
        # Expected format: test_task_name_test-id.json
        parts = filename.split('_', 2)
        if len(parts) < 3:
            continue
            
        # The last part contains test_id.json
        last_part = parts[2]
        test_id_with_ext = last_part.split('.', 1)[0]
        
        # Check if this is a hyphenated test ID format
        if '-' in test_id_with_ext:
            test_id = test_id_with_ext
            task = parts[1]
            
            results[test_id] = {
                "file": file_path,
                "task": task
            }
    
    return results

if __name__ == "__main__":
    # Ensure required environment variables are set
    if not os.getenv("OPENAI_API_KEY") and not os.path.exists(".env"):
        print("Warning: OPENAI_API_KEY environment variable not set and no .env file found.")
        print("LLM evaluation will fail without an OpenAI API key.")
        print("Please set OPENAI_API_KEY or create a .env file with this variable.")
    
    main()