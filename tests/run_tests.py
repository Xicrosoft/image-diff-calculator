#!/usr/bin/env python3
"""
Test runner for Image Difference Calculator
Provides easy access to run all tests with different configurations
"""

import os
import sys
import unittest
import argparse
import subprocess
from pathlib import Path

def run_unit_tests(verbose=True):
    """Run unit tests only"""
    print("🧪 Running Unit Tests...")
    print("=" * 50)

    # Discover and run unit tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_image_diff_calculator.py')

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, buffer=True)
    result = runner.run(suite)

    return result.wasSuccessful()

def run_integration_tests(verbose=True):
    """Run integration tests only"""
    print("\n🔧 Running Integration Tests...")
    print("=" * 50)

    # Discover and run integration tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_integration.py')

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, buffer=True)
    result = runner.run(suite)

    return result.wasSuccessful()

def run_all_tests(verbose=True):
    """Run all tests"""
    print("🚀 Running All Tests...")
    print("=" * 60)

    # Run unit tests first
    unit_success = run_unit_tests(verbose)

    # Run integration tests
    integration_success = run_integration_tests(verbose)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")

    overall_success = unit_success and integration_success
    print(f"Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    return overall_success

def run_coverage_report():
    """Run tests with coverage report"""
    print("📈 Running Tests with Coverage Report...")
    print("=" * 50)

    try:
        # Check if coverage is installed
        subprocess.run([sys.executable, "-c", "import coverage"], check=True, capture_output=True)

        # Run tests with coverage
        test_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(test_dir)

        cmd = [
            sys.executable, "-m", "coverage", "run", "--source", parent_dir,
            "-m", "unittest", "discover", "-s", test_dir, "-p", "test_*.py", "-v"
        ]

        result = subprocess.run(cmd, cwd=parent_dir)

        if result.returncode == 0:
            print("\n📊 Generating Coverage Report...")
            subprocess.run([sys.executable, "-m", "coverage", "report", "-m"], cwd=parent_dir)
            subprocess.run([sys.executable, "-m", "coverage", "html"], cwd=parent_dir)
            print("📄 HTML coverage report generated in htmlcov/")

        return result.returncode == 0

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Coverage module not found. Install with: pip install coverage")
        print("🔄 Running tests without coverage...")
        return run_all_tests()

def run_specific_test(test_name, verbose=True):
    """Run a specific test method or class"""
    print(f"🎯 Running Specific Test: {test_name}")
    print("=" * 50)

    # Try to run the specific test
    try:
        suite = unittest.TestLoader().loadTestsFromName(test_name)
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, buffer=True)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception as e:
        print(f"❌ Error running test {test_name}: {e}")
        return False

def check_test_environment():
    """Check if test environment is properly set up"""
    print("🔍 Checking Test Environment...")
    print("=" * 40)

    # Check if test images exist
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    if not os.path.exists(test_images_dir):
        print("❌ Test images directory not found. Creating test images...")
        try:
            subprocess.run([sys.executable, "create_test_images.py"],
                         cwd=os.path.dirname(__file__), check=True)
            print("✅ Test images created successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to create test images")
            return False
    else:
        print("✅ Test images directory found")

    # Check required modules
    required_modules = ['cv2', 'numpy', 'matplotlib']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} is available")
        except ImportError:
            print(f"❌ {module} is missing")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n✅ Test environment is ready!")
    return True

def main():
    """Main function for test runner"""
    parser = argparse.ArgumentParser(description='Test runner for Image Difference Calculator')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--coverage', action='store_true', help='Run tests with coverage report')
    parser.add_argument('--test', type=str, help='Run specific test (e.g., TestClass.test_method)')
    parser.add_argument('--check', action='store_true', help='Check test environment setup')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output verbosity')

    args = parser.parse_args()

    # Set verbosity
    verbose = not args.quiet

    # Check environment if requested
    if args.check:
        success = check_test_environment()
        sys.exit(0 if success else 1)

    # Ensure test environment is set up before running tests
    if not check_test_environment():
        print("❌ Test environment check failed. Fix issues before running tests.")
        sys.exit(1)

    success = False

    if args.unit:
        success = run_unit_tests(verbose)
    elif args.integration:
        success = run_integration_tests(verbose)
    elif args.coverage:
        success = run_coverage_report()
    elif args.test:
        success = run_specific_test(args.test, verbose)
    else:
        success = run_all_tests(verbose)

    print(f"\n{'🎉 Tests completed successfully!' if success else '💥 Tests failed!'}")
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
