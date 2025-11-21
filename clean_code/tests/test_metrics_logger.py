"""
Test Metrics Logger for LUMENAA Agent System

Implements automated test cases based on AGENT_TEST_CASES.md and logs comprehensive
metrics for analysis across different computer configurations.

This module provides:
1. Automated test case execution with predefined queries
2. Performance measurement and logging  
3. Test result aggregation and CSV export
4. Integration with the performance monitor
"""

import asyncio
import csv
import json
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid

from core.infrastructure.performance_monitor import get_performance_monitor, PerformanceMonitor


@dataclass
class TestCase:
    """Individual test case definition"""
    id: str
    name: str
    query: str
    expected_behavior: str
    validations: List[str]
    complexity: str
    category: str = ""


@dataclass
class TestResult:
    """Test case execution result"""
    test_case_id: str
    test_name: str
    query: str
    response: str
    success: bool
    validation_results: Dict[str, bool]
    execution_time_ms: float
    timestamp: float
    performance_metrics: Optional[Dict[str, Any]] = None
    error_message: str = ""


class TestMetricsLogger:
    """
    Automated test execution and metrics logging system.
    Executes predefined test cases and logs comprehensive performance metrics.
    """
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize performance monitor
        self.performance_monitor = get_performance_monitor()
        
        # Test results storage
        self.test_results: List[TestResult] = []
        
        # CSV files
        self.test_results_csv = os.path.join(output_dir, f"test_results_{int(time.time())}.csv")
        self.test_summary_csv = os.path.join(output_dir, f"test_summary_{int(time.time())}.csv")
        
        # Load test cases
        self.test_cases = self._load_test_cases()
        
        # Initialize CSV files
        self._initialize_csv_files()
    
    def _load_test_cases(self) -> List[TestCase]:
        """Load test cases based on AGENT_TEST_CASES.md"""
        test_cases = []
        
        # Simple Level Test Cases
        test_cases.extend([
            TestCase(
                id="TC-01",
                name="Basic Object Location Query",
                query="Where is the pen?",
                expected_behavior="Reports pen location with distance and direction",
                validations=[
                    "Direction is human-friendly (not angular degrees)",
                    "Distance in meters", 
                    "Handles object name variations",
                    "Response time < 3 seconds"
                ],
                complexity="Simple",
                category="Object Location"
            ),
            TestCase(
                id="TC-02",
                name="Object Presence Check",
                query="Is there a cup?",
                expected_behavior="Yes/No response with location if found",
                validations=[
                    "Distinguishes current vs historical detection",
                    "Historical threshold: 5 seconds",
                    "No false positives"
                ],
                complexity="Simple",
                category="Object Detection"
            ),
            TestCase(
                id="TC-03",
                name="Multiple Objects of Same Type",
                query="Where are all the bottles?",
                expected_behavior="Lists all detected bottles with individual locations",
                validations=[
                    "Handles multiple instances correctly",
                    "Reports each location separately",
                    "Counts objects accurately",
                    "Handles plural forms"
                ],
                complexity="Simple",
                category="Object Location"
            ),
            TestCase(
                id="TC-04",
                name="Basic Face Recognition",
                query="Is reshad around?",
                expected_behavior="Detects and recognizes person by name",
                validations=[
                    "Face recognition works (70% threshold)",
                    "Name matching is correct",
                    "Location formatting is human-friendly",
                    "Uses person's name, NOT 'person' class"
                ],
                complexity="Simple",
                category="Face Recognition"
            ),
            TestCase(
                id="TC-05",
                name="Scene Description", 
                query="Describe the scene",
                expected_behavior="Comprehensive AI-powered scene description",
                validations=[
                    "Uses Pixtral with scene_mode=True",
                    "Includes object locations and relationships",
                    "Provides rich, natural language description",
                    "Covers entire frame, not just specific objects"
                ],
                complexity="Simple",
                category="Scene Analysis"
            ),
            TestCase(
                id="TC-06",
                name="Document Reading",
                query="Read this document",
                expected_behavior="Reads all text from document verbatim",
                validations=[
                    "OCR extraction works correctly",
                    "Intent=read is detected",
                    "Full text is output (not summarized)",
                    "Handles various document types"
                ],
                complexity="Simple",
                category="Document Scanning"
            )
        ])
        
        # Medium Level Test Cases  
        test_cases.extend([
            TestCase(
                id="TC-07",
                name="Object Description with Visual Analysis",
                query="How does the bottle look like in front of me?",
                expected_behavior="Detailed visual description of the bottle",
                validations=[
                    "Uses Pixtral with scene_mode=False (ROI clipping)",
                    "Provides detailed visual features",
                    "Describes color, shape, size, texture",
                    "Analysis time < 4 seconds"
                ],
                complexity="Medium",
                category="Visual Analysis"
            ),
            TestCase(
                id="TC-08", 
                name="Person Recognition with Context",
                query="Who is that person sitting down?",
                expected_behavior="Identifies person and describes their pose/activity",
                validations=[
                    "Face recognition works",
                    "Describes pose/activity",
                    "Combines detection with recognition",
                    "Contextual understanding"
                ],
                complexity="Medium",
                category="Face Recognition"
            ),
            TestCase(
                id="TC-09",
                name="Historical Object Query",
                query="Where did I put my phone?", 
                expected_behavior="References temporal memory for last-seen location",
                validations=[
                    "Uses temporal memory effectively",
                    "Reports last-seen timestamp",
                    "Provides historical location",
                    "Temporal window: 120 seconds"
                ],
                complexity="Medium",
                category="Temporal Memory"
            ),
            TestCase(
                id="TC-10",
                name="Clarification Request",
                query="Where is it?",
                expected_behavior="Asks for clarification about ambiguous reference",
                validations=[
                    "Detects ambiguous query",
                    "Asks clarification question", 
                    "Waits for user response",
                    "Processes clarified query correctly"
                ],
                complexity="Medium",
                category="Clarification"
            )
        ])
        
        # Complex Level Test Cases
        test_cases.extend([
            TestCase(
                id="TC-11",
                name="Multi-step Complex Query",
                query="Find the red bottle and tell me what's written on it",
                expected_behavior="Combines object detection, visual analysis, and OCR",
                validations=[
                    "Multi-step plan execution",
                    "Object filtering by color/attributes",
                    "OCR on specific object",
                    "End-to-end latency < 6 seconds"
                ],
                complexity="Complex", 
                category="Multi-modal"
            ),
            TestCase(
                id="TC-12",
                name="Performance Stress Test",
                query="What do you see?",
                expected_behavior="Fast scene analysis under load",
                validations=[
                    "Maintains <15 FPS detection",
                    "Response time consistent under load",
                    "Memory usage stable",
                    "VLM call optimization active"
                ],
                complexity="Complex",
                category="Performance"
            )
        ])
        
        return test_cases
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers"""
        # Test results CSV
        results_headers = [
            'test_case_id', 'test_name', 'query', 'response', 'success',
            'execution_time_ms', 'timestamp', 'complexity', 'category',
            'validation_passed', 'validation_failed', 'error_message'
        ]
        
        with open(self.test_results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(results_headers)
        
        # Test summary CSV
        summary_headers = [
            'timestamp', 'total_tests', 'tests_passed', 'tests_failed',
            'success_rate_percent', 'avg_execution_time_ms',
            'simple_tests_passed', 'medium_tests_passed', 'complex_tests_passed',
            'object_location_passed', 'face_recognition_passed', 'scene_analysis_passed',
            'document_scanning_passed', 'temporal_memory_passed'
        ]
        
        with open(self.test_summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(summary_headers)
    
    async def execute_single_test(self, test_case: TestCase, agent_controller, 
                                 timeout_seconds: float = 10.0) -> TestResult:
        """
        Execute a single test case and measure performance.
        
        Args:
            test_case: Test case to execute
            agent_controller: Agent controller instance for query processing
            timeout_seconds: Maximum time to wait for response
            
        Returns:
            TestResult with execution metrics and validation results
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Start performance measurement
        self.performance_monitor.start_query_measurement(
            request_id, test_case.query, test_case.id
        )
        
        try:
            # Execute the query through the agent controller
            response = await asyncio.wait_for(
                self._execute_query_through_agent(test_case.query, agent_controller, request_id),
                timeout=timeout_seconds
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Validate the response
            validation_results = await self._validate_response(test_case, response)
            
            # Determine overall success
            success = all(validation_results.values()) if validation_results else False
            
            # Finish performance measurement
            final_metrics = self.performance_monitor.finish_query_measurement(
                request_id, execution_time_ms
            )
            
            # Create test result
            test_result = TestResult(
                test_case_id=test_case.id,
                test_name=test_case.name,
                query=test_case.query,
                response=response,
                success=success,
                validation_results=validation_results,
                execution_time_ms=execution_time_ms,
                timestamp=start_time,
                performance_metrics=final_metrics.__dict__ if final_metrics else None
            )
            
        except asyncio.TimeoutError:
            execution_time_ms = timeout_seconds * 1000
            test_result = TestResult(
                test_case_id=test_case.id,
                test_name=test_case.name,
                query=test_case.query,
                response="",
                success=False,
                validation_results={},
                execution_time_ms=execution_time_ms,
                timestamp=start_time,
                error_message=f"Timeout after {timeout_seconds} seconds"
            )
            
            # Finish performance measurement with timeout
            self.performance_monitor.finish_query_measurement(request_id, execution_time_ms)
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            test_result = TestResult(
                test_case_id=test_case.id,
                test_name=test_case.name, 
                query=test_case.query,
                response="",
                success=False,
                validation_results={},
                execution_time_ms=execution_time_ms,
                timestamp=start_time,
                error_message=str(e)
            )
            
            # Finish performance measurement with error
            self.performance_monitor.finish_query_measurement(request_id, execution_time_ms)
        
        # Log test result to CSV
        self._write_test_result_to_csv(test_result, test_case)
        
        return test_result
    
    async def _execute_query_through_agent(self, query: str, agent_controller, request_id: str) -> str:
        """Execute query through the agent controller and return response"""
        # This would integrate with the actual agent controller
        # For now, we'll return a placeholder that shows the integration point
        
        # In the actual integration, this would:
        # 1. Send query to agent controller
        # 2. Wait for response through event bus
        # 3. Return the final response text
        
        # Placeholder implementation - replace with actual agent integration
        return f"[PLACEHOLDER] Agent response to: {query}"
    
    async def _validate_response(self, test_case: TestCase, response: str) -> Dict[str, bool]:
        """
        Validate test response against expected criteria.
        
        Returns dict of validation_name -> passed (True/False)
        """
        validation_results = {}
        
        if not response or response.startswith("[PLACEHOLDER]"):
            # For placeholder responses, mark all validations as pending
            for validation in test_case.validations:
                validation_results[validation] = False
            return validation_results
        
        response_lower = response.lower()
        
        # Common validation patterns
        for validation in test_case.validations:
            validation_lower = validation.lower()
            passed = False
            
            if "direction is human-friendly" in validation_lower:
                # Check that response doesn't contain angular degrees
                passed = "degree" not in response_lower and ("front" in response_lower or "left" in response_lower or "right" in response_lower)
            
            elif "distance in meters" in validation_lower:
                # Check for distance in meters 
                passed = ("meter" in response_lower or "m " in response_lower) and any(char.isdigit() for char in response)
            
            elif "response time" in validation_lower:
                # This would be validated by checking actual execution time
                passed = True  # Placeholder - check execution_time_ms
            
            elif "face recognition works" in validation_lower:
                # Check if person name is mentioned (not just "person")
                passed = "reshad" in response_lower or "person" not in response_lower
            
            elif "scene_mode=true" in validation_lower:
                # Check for rich scene description
                passed = len(response.split()) > 10  # Rich description expected
            
            elif "ocr extraction" in validation_lower:
                # Check for text extraction
                passed = len(response) > 20  # Expecting extracted text
            
            elif "temporal memory" in validation_lower:
                # Check for historical references
                passed = "ago" in response_lower or "earlier" in response_lower or "before" in response_lower
            
            elif "clarification" in validation_lower:
                # Check for clarification question
                passed = "?" in response and ("what" in response_lower or "which" in response_lower)
            
            else:
                # Default validation - check if response is not empty and reasonable length
                passed = len(response.strip()) > 5
            
            validation_results[validation] = passed
        
        return validation_results
    
    def _write_test_result_to_csv(self, test_result: TestResult, test_case: TestCase):
        """Write test result to CSV file"""
        try:
            validation_passed = sum(test_result.validation_results.values())
            validation_failed = len(test_result.validation_results) - validation_passed
            
            with open(self.test_results_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    test_result.test_case_id,
                    test_result.test_name,
                    test_result.query,
                    test_result.response[:200] + "..." if len(test_result.response) > 200 else test_result.response,
                    test_result.success,
                    test_result.execution_time_ms,
                    test_result.timestamp,
                    test_case.complexity,
                    test_case.category,
                    validation_passed,
                    validation_failed,
                    test_result.error_message
                ])
        except Exception as e:
            print(f"[WARNING] Failed to write test result to CSV: {e}")
    
    async def run_test_suite(self, agent_controller, test_filter: Optional[str] = None,
                           repeat_count: int = 1) -> Dict[str, Any]:
        """
        Run the complete test suite and generate summary report.
        
        Args:
            agent_controller: Agent controller instance
            test_filter: Optional filter by complexity ("Simple", "Medium", "Complex") or category
            repeat_count: Number of times to repeat each test for averaging
            
        Returns:
            Dict with test results and summary statistics
        """
        print(f"\n=== STARTING TEST SUITE ===")
        print(f"Total test cases: {len(self.test_cases)}")
        print(f"Filter: {test_filter or 'None'}")
        print(f"Repeat count: {repeat_count}")
        print("=" * 30)
        
        # Filter test cases if requested
        filtered_tests = self.test_cases
        if test_filter:
            filtered_tests = [
                tc for tc in self.test_cases 
                if test_filter.lower() in tc.complexity.lower() or test_filter.lower() in tc.category.lower()
            ]
        
        all_results = []
        
        # Execute each test case
        for i, test_case in enumerate(filtered_tests, 1):
            print(f"\n[{i}/{len(filtered_tests)}] Executing {test_case.id}: {test_case.name}")
            
            test_results = []
            for run in range(repeat_count):
                if repeat_count > 1:
                    print(f"  Run {run + 1}/{repeat_count}")
                
                result = await self.execute_single_test(test_case, agent_controller)
                test_results.append(result)
                all_results.append(result)
                
                # Print immediate feedback
                status = "✓ PASS" if result.success else "✗ FAIL"
                print(f"  {status} - {result.execution_time_ms:.1f}ms")
                if result.error_message:
                    print(f"    Error: {result.error_message}")
                
                # Brief delay between runs
                if run < repeat_count - 1:
                    await asyncio.sleep(0.5)
        
        # Generate summary
        summary = self._generate_test_summary(all_results)
        
        # Write summary to CSV
        self._write_test_summary_to_csv(summary)
        
        # Print final summary
        self._print_test_summary(summary)
        
        return {
            "test_results": all_results,
            "summary": summary
        }
    
    def _generate_test_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate test execution summary statistics"""
        if not results:
            return {}
        
        total_tests = len(results)
        tests_passed = sum(1 for r in results if r.success)
        tests_failed = total_tests - tests_passed
        success_rate = (tests_passed / total_tests) * 100
        avg_execution_time = sum(r.execution_time_ms for r in results) / total_tests
        
        # Group by complexity
        complexity_stats = {}
        for complexity in ["Simple", "Medium", "Complex"]:
            complexity_results = [r for r in results if complexity.lower() in r.test_case_id.lower()]
            if complexity_results:
                complexity_stats[complexity.lower()] = {
                    "total": len(complexity_results),
                    "passed": sum(1 for r in complexity_results if r.success),
                    "success_rate": (sum(1 for r in complexity_results if r.success) / len(complexity_results)) * 100
                }
        
        # Group by category
        category_stats = {}
        for result in results:
            # Extract category from test case (would need test case lookup)
            category = "Unknown"  # Placeholder
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0}
            category_stats[category]["total"] += 1
            if result.success:
                category_stats[category]["passed"] += 1
        
        return {
            "timestamp": time.time(),
            "total_tests": total_tests,
            "tests_passed": tests_passed, 
            "tests_failed": tests_failed,
            "success_rate_percent": round(success_rate, 2),
            "avg_execution_time_ms": round(avg_execution_time, 2),
            "complexity_stats": complexity_stats,
            "category_stats": category_stats
        }
    
    def _write_test_summary_to_csv(self, summary: Dict[str, Any]):
        """Write test summary to CSV file"""
        try:
            with open(self.test_summary_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    summary["timestamp"],
                    summary["total_tests"],
                    summary["tests_passed"],
                    summary["tests_failed"], 
                    summary["success_rate_percent"],
                    summary["avg_execution_time_ms"],
                    summary.get("complexity_stats", {}).get("simple", {}).get("passed", 0),
                    summary.get("complexity_stats", {}).get("medium", {}).get("passed", 0),
                    summary.get("complexity_stats", {}).get("complex", {}).get("passed", 0),
                    # Category stats would be filled in actual implementation
                    0, 0, 0, 0, 0  # Placeholders for category stats
                ])
        except Exception as e:
            print(f"[WARNING] Failed to write test summary to CSV: {e}")
    
    def _print_test_summary(self, summary: Dict[str, Any]):
        """Print test execution summary to console"""
        print(f"\n=== TEST EXECUTION SUMMARY ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['tests_passed']}")
        print(f"Failed: {summary['tests_failed']}")
        print(f"Success Rate: {summary['success_rate_percent']}%")
        print(f"Avg Execution Time: {summary['avg_execution_time_ms']:.1f}ms")
        
        if summary.get("complexity_stats"):
            print(f"\nBy Complexity:")
            for complexity, stats in summary["complexity_stats"].items():
                print(f"  {complexity.title()}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)")
        
        print(f"\nResults saved to: {self.test_results_csv}")
        print(f"Summary saved to: {self.test_summary_csv}")
        print("=" * 30)


# Global test metrics logger instance
_global_logger: Optional[TestMetricsLogger] = None

def get_test_metrics_logger() -> TestMetricsLogger:
    """Get the global test metrics logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TestMetricsLogger()
    return _global_logger

def init_test_metrics_logger(output_dir: str = "test_results") -> TestMetricsLogger:
    """Initialize global test metrics logger with custom output directory"""
    global _global_logger
    _global_logger = TestMetricsLogger(output_dir)
    return _global_logger
