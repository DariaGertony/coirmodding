"""
Script to run all tests and generate comprehensive report

This script runs the complete test suite for the refactored CoIR architecture
and generates detailed reports on test coverage, performance, and validation status.
"""

import pytest
import sys
import os
import subprocess
import time
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print(f"{'-' * 60}")


def run_test_module(module_path, description):
    """Run a specific test module and return results"""
    print(f"\n🧪 Running {description}...")
    print(f"   Module: {module_path}")
    
    start_time = time.time()
    
    # Run pytest for the specific module
    result = pytest.main([
        module_path,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure for quick feedback
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    status = "✅ PASSED" if result == 0 else "❌ FAILED"
    print(f"   Status: {status} ({duration:.2f}s)")
    
    return result == 0, duration


def run_coverage_analysis():
    """Run test coverage analysis"""
    print_section("Coverage Analysis")
    
    test_modules = [
        "tests/test_llm_components.py",
        "tests/test_lexical_search.py", 
        "tests/test_hybrid_search.py",
        "tests/test_evaluation_integration.py",
        "tests/test_main_evaluation.py",
        "tests/test_full_integration.py",
        "tests/test_performance.py",
        "tests/test_backward_compatibility.py",
        "tests/test_e2e_workflows.py",
        "tests/test_configuration_system.py",
        "tests/test_documentation.py"
    ]
    
    print("🔍 Running coverage analysis...")
    
    # Run tests with coverage
    coverage_result = pytest.main([
        "--cov=coir",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
        *test_modules
    ])
    
    if coverage_result == 0:
        print("✅ Coverage analysis completed successfully")
        print("📊 Coverage reports generated:")
        print("   - HTML report: htmlcov/index.html")
        print("   - JSON report: coverage.json")
    else:
        print("❌ Coverage analysis failed")
    
    return coverage_result == 0


def check_dependencies():
    """Check if all required dependencies are available"""
    print_section("Dependency Check")
    
    required_packages = [
        "pytest",
        "pytest-cov",
        "numpy",
        "torch",
        "psutil"  # For performance tests
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All dependencies available")
    return True


def run_all_tests():
    """Run complete test suite"""
    print_header("CoIR Refactoring Validation Test Suite")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot proceed without required dependencies")
        return False
    
    print_section("Test Execution Plan")
    
    test_modules = [
        ("tests/test_llm_components.py", "LLM Components Tests"),
        ("tests/test_lexical_search.py", "Lexical Search Tests"), 
        ("tests/test_hybrid_search.py", "Hybrid Search Tests"),
        ("tests/test_evaluation_integration.py", "Evaluation Integration Tests"),
        ("tests/test_main_evaluation.py", "Main Evaluation Tests"),
        ("tests/test_full_integration.py", "Full Integration Tests"),
        ("tests/test_performance.py", "Performance Benchmarks"),
        ("tests/test_backward_compatibility.py", "Backward Compatibility Tests"),
        ("tests/test_e2e_workflows.py", "End-to-End Workflow Tests"),
        ("tests/test_configuration_system.py", "Configuration System Tests"),
        ("tests/test_documentation.py", "Documentation Validation Tests")
    ]
    
    print(f"📋 Planning to run {len(test_modules)} test modules:")
    for module_path, description in test_modules:
        status = "✅" if os.path.exists(module_path) else "❌"
        print(f"   {status} {description}")
    
    print_section("Test Execution")
    
    results = []
    total_duration = 0
    
    for module_path, description in test_modules:
        if not os.path.exists(module_path):
            print(f"⚠️  Skipping {description} - file not found: {module_path}")
            results.append((description, False, 0))
            continue
        
        success, duration = run_test_module(module_path, description)
        results.append((description, success, duration))
        total_duration += duration
        
        if not success:
            print(f"❌ {description} failed - stopping execution")
            break
    
    print_section("Test Results Summary")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} modules passed")
    print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
    
    for description, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} {description} ({duration:.2f}s)")
    
    # Run coverage analysis if all tests passed
    if passed == total:
        coverage_success = run_coverage_analysis()
        
        print_section("Final Status")
        
        if coverage_success:
            print("🎉 ALL TESTS PASSED WITH COVERAGE ANALYSIS!")
            print("✅ Refactoring validation completed successfully")
            print("\n📋 Next Steps:")
            print("   1. Review coverage report: htmlcov/index.html")
            print("   2. Check validation checklist: VALIDATION_CHECKLIST.md")
            print("   3. Proceed with deployment/integration")
        else:
            print("⚠️  Tests passed but coverage analysis failed")
            return False
    else:
        print_section("Final Status")
        print("❌ SOME TESTS FAILED")
        print("🔧 Please fix failing tests before proceeding")
        return False
    
    return True


def run_quick_tests():
    """Run a quick subset of tests for rapid feedback"""
    print_header("CoIR Quick Test Suite")
    
    quick_test_modules = [
        ("tests/test_main_evaluation.py", "Main Evaluation Tests"),
        ("tests/test_backward_compatibility.py", "Backward Compatibility Tests"),
        ("tests/test_configuration_system.py", "Configuration System Tests")
    ]
    
    print("🚀 Running quick test suite for rapid feedback...")
    
    results = []
    total_duration = 0
    
    for module_path, description in quick_test_modules:
        if not os.path.exists(module_path):
            print(f"⚠️  Skipping {description} - file not found")
            continue
        
        success, duration = run_test_module(module_path, description)
        results.append((description, success, duration))
        total_duration += duration
        
        if not success:
            break
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n📊 Quick Test Results: {passed}/{total} modules passed ({total_duration:.2f}s)")
    
    if passed == total:
        print("✅ Quick tests passed! Ready for full test suite.")
    else:
        print("❌ Quick tests failed. Fix issues before running full suite.")
    
    return passed == total


def run_specific_category(category):
    """Run tests for a specific category"""
    categories = {
        "integration": [
            ("tests/test_full_integration.py", "Full Integration Tests"),
            ("tests/test_evaluation_integration.py", "Evaluation Integration Tests"),
            ("tests/test_e2e_workflows.py", "End-to-End Workflow Tests")
        ],
        "compatibility": [
            ("tests/test_backward_compatibility.py", "Backward Compatibility Tests"),
            ("tests/test_documentation.py", "Documentation Validation Tests")
        ],
        "performance": [
            ("tests/test_performance.py", "Performance Benchmarks")
        ],
        "components": [
            ("tests/test_llm_components.py", "LLM Components Tests"),
            ("tests/test_lexical_search.py", "Lexical Search Tests"),
            ("tests/test_hybrid_search.py", "Hybrid Search Tests"),
            ("tests/test_configuration_system.py", "Configuration System Tests")
        ]
    }
    
    if category not in categories:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return False
    
    print_header(f"CoIR {category.title()} Tests")
    
    test_modules = categories[category]
    results = []
    total_duration = 0
    
    for module_path, description in test_modules:
        if not os.path.exists(module_path):
            print(f"⚠️  Skipping {description} - file not found")
            continue
        
        success, duration = run_test_module(module_path, description)
        results.append((description, success, duration))
        total_duration += duration
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n📊 {category.title()} Test Results: {passed}/{total} modules passed ({total_duration:.2f}s)")
    
    return passed == total


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            success = run_quick_tests()
        elif command == "coverage":
            success = run_coverage_analysis()
        elif command in ["integration", "compatibility", "performance", "components"]:
            success = run_specific_category(command)
        elif command == "help":
            print("CoIR Test Runner")
            print("\nUsage:")
            print("  python run_all_tests.py           # Run full test suite")
            print("  python run_all_tests.py quick     # Run quick tests")
            print("  python run_all_tests.py coverage  # Run coverage analysis only")
            print("  python run_all_tests.py integration    # Run integration tests")
            print("  python run_all_tests.py compatibility  # Run compatibility tests")
            print("  python run_all_tests.py performance    # Run performance tests")
            print("  python run_all_tests.py components     # Run component tests")
            print("  python run_all_tests.py help      # Show this help")
            return
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python run_all_tests.py help' for usage information")
            sys.exit(1)
    else:
        success = run_all_tests()
    
    if success:
        print("\n🎉 Test execution completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()