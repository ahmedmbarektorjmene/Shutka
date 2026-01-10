"""
Test Suite 3: Algorithmic Thinking
Tests multi-step reasoning for complex algorithms
"""
import os
import json
from typing import List, Dict, Tuple


class AlgorithmicTestSuite:
    """
    Test suite for algorithmic thinking evaluation
    """
    def __init__(self, test_suite_dir: str = "evaluation/test_suites"):
        self.test_suite_dir = test_suite_dir
        os.makedirs(test_suite_dir, exist_ok=True)
        self.tests = self._load_tests()
    
    def _load_tests(self) -> List[Dict]:
        """Load algorithmic tests from file or create default tests"""
        test_file = os.path.join(self.test_suite_dir, "algorithmic_tests.json")
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                return json.load(f)
        
        # Default test cases
        default_tests = [
            {
                "id": "algo_001",
                "prompt": "Implement MergeSort in TypeScript. Write a function `mergeSort(arr: number[]): number[]` that sorts an array using the merge sort algorithm.",
                "test_cases": [
                    {
                        "input": "[64, 34, 25, 12, 22, 11, 90]",
                        "expected": "[11, 12, 22, 25, 34, 64, 90]",
                        "description": "Basic merge sort"
                    },
                    {
                        "input": "[5, 2, 8, 1, 9]",
                        "expected": "[1, 2, 5, 8, 9]",
                        "description": "Small array"
                    }
                ],
                "efficiency_check": "O(n log n) time complexity"
            },
            {
                "id": "algo_002",
                "prompt": "Implement Binary Search in TypeScript. Write a function `binarySearch(arr: number[], target: number): number` that returns the index of target in a sorted array, or -1 if not found.",
                "test_cases": [
                    {
                        "input": "binarySearch([1, 2, 3, 4, 5], 3)",
                        "expected": "2",
                        "description": "Find existing element"
                    },
                    {
                        "input": "binarySearch([1, 2, 3, 4, 5], 6)",
                        "expected": "-1",
                        "description": "Element not found"
                    }
                ],
                "efficiency_check": "O(log n) time complexity"
            },
            {
                "id": "algo_003",
                "prompt": "Implement a function `twoSum(nums: number[], target: number): number[]` that finds two numbers in an array that add up to target. Return the indices of the two numbers.",
                "test_cases": [
                    {
                        "input": "twoSum([2, 7, 11, 15], 9)",
                        "expected": "[0, 1]",
                        "description": "Basic two sum"
                    },
                    {
                        "input": "twoSum([3, 2, 4], 6)",
                        "expected": "[1, 2]",
                        "description": "Different indices"
                    }
                ],
                "efficiency_check": "O(n) time complexity preferred"
            },
            {
                "id": "algo_004",
                "prompt": "Implement a function `longestCommonSubsequence(s1: string, s2: string): number` that returns the length of the longest common subsequence between two strings.",
                "test_cases": [
                    {
                        "input": "longestCommonSubsequence('ABCDGH', 'AEDFHR')",
                        "expected": "3",
                        "description": "LCS example"
                    },
                    {
                        "input": "longestCommonSubsequence('AGGTAB', 'GXTXAYB')",
                        "expected": "4",
                        "description": "Another LCS example"
                    }
                ],
                "efficiency_check": "Dynamic programming approach"
            },
            {
                "id": "algo_005",
                "prompt": "Implement a function `isValidSudoku(board: string[][]): boolean` that validates a 9x9 Sudoku board.",
                "test_cases": [
                    {
                        "input": "isValidSudoku([['5','3','.','.','7','.','.','.','.'],['6','.','.','1','9','5','.','.','.'],['.','9','8','.','.','.','.','6','.'],['8','.','.','.','6','.','.','.','3'],['4','.','.','8','.','3','.','.','1'],['7','.','.','.','2','.','.','.','6'],['.','6','.','.','.','.','2','8','.'],['.','.','.','4','1','9','.','.','5'],['.','.','.','.','8','.','.','7','9']])",
                        "expected": "true",
                        "description": "Valid Sudoku board"
                    }
                ],
                "efficiency_check": "Efficient validation"
            },
            {
                "id": "algo_006",
                "prompt": "Implement a function `maxSubarraySum(arr: number[]): number` that finds the maximum sum of a contiguous subarray (Kadane's algorithm).",
                "test_cases": [
                    {
                        "input": "maxSubarraySum([-2, 1, -3, 4, -1, 2, 1, -5, 4])",
                        "expected": "6",
                        "description": "Maximum subarray sum"
                    },
                    {
                        "input": "maxSubarraySum([-1, -2, -3])",
                        "expected": "-1",
                        "description": "All negative numbers"
                    }
                ],
                "efficiency_check": "O(n) time complexity"
            }
        ]
        
        # Save default tests
        with open(test_file, 'w') as f:
            json.dump(default_tests, f, indent=2)
        
        return default_tests
    
    def get_tests(self) -> List[Dict]:
        """Get all test cases"""
        return self.tests
    
    def create_test_harness(self, function_code: str, test_case: Dict) -> str:
        """
        Create a test harness for running a test case
        """
        test_code = f"""
{function_code}

// Test case
const result = {test_case['input']};
const expected = {test_case['expected']};

// Comparison function
function arrayEquals(a: any, b: any): boolean {{
    if (a === b) return true;
    if (Array.isArray(a) && Array.isArray(b)) {{
        if (a.length !== b.length) return false;
        return a.every((val, idx) => arrayEquals(val, b[idx]));
    }}
    return false;
}}

const passed = arrayEquals(result, expected);
console.log(passed ? 'PASS' : 'FAIL');
if (!passed) {{
    console.log('Expected:', expected);
    console.log('Got:', result);
}}
"""
        return test_code
    
    def run_test(self, function_code: str, test_case: Dict) -> Tuple[bool, str]:
        """
        Run a test case and return (passed, output)
        """
        import tempfile
        import subprocess
        
        test_harness = self.create_test_harness(function_code, test_case)
        
        # Create temporary file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False, encoding='utf-8', errors='replace') as f:
            f.write(test_harness)
            temp_path = f.name
        
        try:
            # Try to run with bun
            result = subprocess.run(
                ['bun', 'run', temp_path],
                capture_output=True,
                text=True,
                timeout=1000
            )
            
            if result.returncode == 0 and 'PASS' in result.stdout:
                return True, result.stdout
            else:
                return False, result.stderr or result.stdout
        except FileNotFoundError:
            # Try tsx
            try:
                result = subprocess.run(
                    ['bunx', 'tsx', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=1000
                )
                if result.returncode == 0 and 'PASS' in result.stdout:
                    return True, result.stdout
                else:
                    return False, result.stderr or result.stdout
            except:
                return False, "No TypeScript runtime available"
        except Exception as e:
            return False, str(e)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def evaluate_efficiency(self, code: str, efficiency_check: str) -> int:
        """
        Evaluate code efficiency (0=incorrect, 1=partial, 2=correct & efficient)
        This is a simplified check - in practice, you'd analyze the code structure
        """
        # Basic heuristics for efficiency
        code_lower = code.lower()
        
        if 'for' in code_lower and 'for' in code_lower:
            # Nested loops might indicate O(n^2) or worse
            if 'log' in efficiency_check.lower() and 'n log' in efficiency_check.lower():
                return 1  # Partial - might not be optimal
        elif 'while' in code_lower or 'recursion' in code_lower or 'recursive' in code_lower:
            if 'log' in efficiency_check.lower():
                return 2  # Likely efficient
        elif 'map' in code_lower or 'filter' in code_lower:
            return 1  # Functional approach, might be efficient
        
        # Default: assume correct if tests pass
        return 2
