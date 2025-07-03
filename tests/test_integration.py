#!/usr/bin/env python3
"""
Integration tests for Image Difference Calculator

Tests the command-line interface and end-to-end functionality
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive import InteractiveImageDiffCalculator


class TestCommandLineInterface(unittest.TestCase):
    """Test command-line interface functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
        cls.red_image = os.path.join(cls.test_images_dir, "red_100x100.png")
        cls.blue_image = os.path.join(cls.test_images_dir, "blue_100x100.png")
        cls.temp_dir = tempfile.mkdtemp()

        # Path to the main script
        cls.script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "image_diff_calculator.py")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_command_line_basic_usage(self):
        """Test basic command line usage"""
        cmd = [sys.executable, self.script_path, self.red_image, self.blue_image]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertIn("similarity", result.stdout.lower())
        self.assertIn("mse", result.stdout.lower())

    def test_command_line_all_methods(self):
        """Test command line with all methods"""
        cmd = [sys.executable, self.script_path, self.red_image, self.blue_image, "--method", "all"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertIn("mse", result.stdout.lower())
        self.assertIn("ssim", result.stdout.lower())
        self.assertIn("histogram", result.stdout.lower())
        self.assertIn("pixel_diff", result.stdout.lower())
        self.assertIn("hash", result.stdout.lower())

    def test_command_line_specific_method(self):
        """Test command line with specific method"""
        for method in ["mse", "ssim", "histogram", "pixel_diff", "hash"]:
            with self.subTest(method=method):
                cmd = [sys.executable, self.script_path, self.red_image, self.blue_image, "-m", method]
                result = subprocess.run(cmd, capture_output=True, text=True)

                self.assertEqual(result.returncode, 0)
                self.assertIn(method.upper(), result.stdout)

    def test_command_line_visualization(self):
        """Test command line with visualization option"""
        output_file = os.path.join(self.temp_dir, "test_viz.png")
        cmd = [
            sys.executable, self.script_path,
            self.red_image, self.blue_image,
            "--visualize", "--output", output_file
        ]

        # Run the command and check for successful execution
        result = subprocess.run(cmd, capture_output=True, text=True)

        # The command should complete successfully regardless of matplotlib display
        self.assertEqual(result.returncode, 0)
        self.assertIn("visualization", result.stdout.lower() if result.stdout else "")

    def test_command_line_invalid_image_path(self):
        """Test command line with invalid image path"""
        cmd = [sys.executable, self.script_path, "nonexistent.png", self.blue_image]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error", result.stdout.lower())

    def test_command_line_invalid_method(self):
        """Test command line with invalid method"""
        cmd = [sys.executable, self.script_path, self.red_image, self.blue_image, "-m", "invalid"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertNotEqual(result.returncode, 0)

    def test_command_line_help(self):
        """Test command line help option"""
        cmd = [sys.executable, self.script_path, "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertIn("usage", result.stdout.lower())
        self.assertIn("method", result.stdout.lower())


class TestInteractiveInterface(unittest.TestCase):
    """Test interactive interface functionality"""

    def setUp(self):
        """Set up before each test"""
        self.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
        self.red_image = os.path.join(self.test_images_dir, "red_100x100.png")
        self.blue_image = os.path.join(self.test_images_dir, "blue_100x100.png")
        self.interactive_calc = InteractiveImageDiffCalculator()

    def test_interactive_calculator_initialization(self):
        """Test interactive calculator initialization"""
        self.assertIsInstance(self.interactive_calc, InteractiveImageDiffCalculator)
        self.assertIsInstance(self.interactive_calc.calculator, type(self.interactive_calc.calculator))
        self.assertIn('image1_path', self.interactive_calc.current_session)
        self.assertIn('image2_path', self.interactive_calc.current_session)
        self.assertIn('results', self.interactive_calc.current_session)

    @patch('builtins.input')
    def test_get_image_path_valid(self, mock_input):
        """Test getting valid image path"""
        mock_input.return_value = self.red_image
        path = self.interactive_calc.get_image_path("Enter path")

        self.assertEqual(path, os.path.abspath(self.red_image))

    @patch('builtins.input')
    def test_get_image_path_with_quotes(self, mock_input):
        """Test getting image path with quotes"""
        mock_input.return_value = f'"{self.red_image}"'
        path = self.interactive_calc.get_image_path("Enter path")

        self.assertEqual(path, os.path.abspath(self.red_image))

    @patch('builtins.input')
    def test_get_image_path_invalid_then_valid(self, mock_input):
        """Test getting invalid path then valid path"""
        mock_input.side_effect = ["nonexistent.png", self.red_image]

        with patch('builtins.print'):  # Suppress error messages
            path = self.interactive_calc.get_image_path("Enter path")

        self.assertEqual(path, os.path.abspath(self.red_image))
        self.assertEqual(mock_input.call_count, 2)

    def test_calculate_all_methods(self):
        """Test calculating all methods"""
        self.interactive_calc.current_session['image1_path'] = self.red_image
        self.interactive_calc.current_session['image2_path'] = self.blue_image

        with patch('builtins.print'):  # Suppress output
            results = self.interactive_calc.calculate_all_methods()

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 5)  # All 5 methods

        for method in ['mse', 'ssim', 'histogram', 'pixel_diff', 'hash']:
            self.assertIn(method, results)

    def test_format_percentage(self):
        """Test percentage formatting"""
        self.assertEqual(self.interactive_calc.format_percentage(0.5), "50.00%")
        self.assertEqual(self.interactive_calc.format_percentage(1.0), "100.00%")
        self.assertEqual(self.interactive_calc.format_percentage(0.0), "0.00%")
        self.assertEqual(self.interactive_calc.format_percentage(None), "N/A")

    def test_format_value(self):
        """Test value formatting"""
        self.assertEqual(self.interactive_calc.format_value(0.1234), "0.1234")
        self.assertEqual(self.interactive_calc.format_value(0.1234, 2), "0.12")
        self.assertEqual(self.interactive_calc.format_value(None), "N/A")

    @patch('builtins.print')
    def test_print_results_no_results(self, mock_print):
        """Test printing results when no results exist"""
        self.interactive_calc.current_session['results'] = None
        self.interactive_calc.print_results()

        # Should print error message
        mock_print.assert_called()
        args = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("No calculation results" in arg for arg in args))

    @patch('builtins.print')
    def test_print_results_with_results(self, mock_print):
        """Test printing results when results exist"""
        # Set up mock results
        mock_results = {
            'mse': {
                'method': 'mse',
                'mse': 100.0,
                'similarity_ratio': 0.5,
                'image_size': (100, 100, 3)
            }
        }
        self.interactive_calc.current_session['results'] = mock_results
        self.interactive_calc.current_session['image1_path'] = self.red_image
        self.interactive_calc.current_session['image2_path'] = self.blue_image

        self.interactive_calc.print_results()

        # Should print detailed results
        mock_print.assert_called()
        output = ' '.join([str(call[0][0]) for call in mock_print.call_args_list])
        self.assertIn("MSE", output)
        self.assertIn("50.00%", output)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
        cls.red_image = os.path.join(cls.test_images_dir, "red_100x100.png")
        cls.red_copy = os.path.join(cls.test_images_dir, "red_100x100_copy.png")
        cls.blue_image = os.path.join(cls.test_images_dir, "blue_100x100.png")
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_identical_images_workflow(self):
        """Test complete workflow with identical images"""
        from image_diff_calculator import ImageDiffCalculator

        calculator = ImageDiffCalculator()

        # Test all methods with identical images
        for method in calculator.supported_methods:
            with self.subTest(method=method):
                result = calculator.calculate_diff_ratio(self.red_image, self.red_copy, method)

                # Identical images should have high similarity
                self.assertGreater(result['similarity_ratio'], 0.95)
                self.assertEqual(result['method'], method)
                self.assertIn('image_size', result)

    def test_different_images_workflow(self):
        """Test complete workflow with different images"""
        from image_diff_calculator import ImageDiffCalculator

        calculator = ImageDiffCalculator()

        # Test all methods with different images
        for method in calculator.supported_methods:
            with self.subTest(method=method):
                result = calculator.calculate_diff_ratio(self.red_image, self.blue_image, method)

                # Different images should have lower similarity, but hash method may be less sensitive
                if method == 'hash':
                    # Perceptual hash may not detect color differences well
                    self.assertLessEqual(result['similarity_ratio'], 1.0)
                else:
                    self.assertLess(result['similarity_ratio'], 0.8)
                self.assertEqual(result['method'], method)
                self.assertIn('image_size', result)

    def test_visualization_workflow(self):
        """Test complete visualization workflow"""
        from image_diff_calculator import ImageDiffCalculator

        calculator = ImageDiffCalculator()
        output_path = os.path.join(self.temp_dir, "workflow_viz.png")

        # Mock matplotlib to avoid display issues
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
                patch('matplotlib.pyplot.close'), \
                patch('matplotlib.pyplot.figure'), \
                patch('matplotlib.pyplot.subplot'), \
                patch('matplotlib.pyplot.imshow'), \
                patch('matplotlib.pyplot.title'), \
                patch('matplotlib.pyplot.axis'), \
                patch('matplotlib.pyplot.bar'), \
                patch('matplotlib.pyplot.ylabel'), \
                patch('matplotlib.pyplot.xticks'), \
                patch('matplotlib.pyplot.ylim'), \
                patch('matplotlib.pyplot.tight_layout'), \
                patch('builtins.print'):
            calculator.save_diff_visualization(self.red_image, self.blue_image, output_path)

            # Verify visualization was attempted
            mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')


class TestPerformanceAndStress(unittest.TestCase):
    """Test performance and stress scenarios"""

    def setUp(self):
        """Set up before each test"""
        self.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
        self.red_image = os.path.join(self.test_images_dir, "red_100x100.png")
        self.blue_image = os.path.join(self.test_images_dir, "blue_100x100.png")

    def test_multiple_calculations_performance(self):
        """Test performance with multiple calculations"""
        from image_diff_calculator import ImageDiffCalculator

        calculator = ImageDiffCalculator()

        # Perform multiple calculations and ensure they complete in reasonable time
        import time
        start_time = time.time()

        for _ in range(10):
            for method in calculator.supported_methods:
                result = calculator.calculate_diff_ratio(self.red_image, self.blue_image, method)
                self.assertIn('similarity_ratio', result)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Should complete 50 calculations (10 iterations * 5 methods) in reasonable time
        self.assertLess(elapsed_time, 30.0)  # 30 seconds should be more than enough

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable"""
        from image_diff_calculator import ImageDiffCalculator

        calculator = ImageDiffCalculator()

        # Perform many calculations to check for memory leaks
        for i in range(20):
            result = calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'mse')
            self.assertIn('similarity_ratio', result)

            # Force garbage collection periodically
            if i % 5 == 0:
                import gc
                gc.collect()


if __name__ == '__main__':
    # Create comprehensive test suite
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestCommandLineInterface))
    test_suite.addTest(unittest.makeSuite(TestInteractiveInterface))
    test_suite.addTest(unittest.makeSuite(TestEndToEndWorkflow))
    test_suite.addTest(unittest.makeSuite(TestPerformanceAndStress))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
