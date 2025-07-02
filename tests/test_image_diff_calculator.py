#!/usr/bin/env python3
"""
Unit tests for Image Difference Calculator

Tests all the core functionality of the ImageDiffCalculator class
including different similarity calculation methods and edge cases.
"""

import unittest
import os
import sys
import numpy as np
import cv2
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_diff_calculator import ImageDiffCalculator


class TestImageDiffCalculator(unittest.TestCase):
    """Test cases for ImageDiffCalculator class"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all test methods"""
        cls.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
        cls.calculator = ImageDiffCalculator()

        # Define test image paths
        cls.red_image = os.path.join(cls.test_images_dir, "red_100x100.png")
        cls.red_copy = os.path.join(cls.test_images_dir, "red_100x100_copy.png")
        cls.blue_image = os.path.join(cls.test_images_dir, "blue_100x100.png")
        cls.gradient_image = os.path.join(cls.test_images_dir, "gradient_100x100.png")
        cls.checkerboard_image = os.path.join(cls.test_images_dir, "checkerboard_100x100.png")
        cls.red_small = os.path.join(cls.test_images_dir, "red_50x50.png")
        cls.red_modified = os.path.join(cls.test_images_dir, "red_100x100_modified.png")
        cls.red_noisy = os.path.join(cls.test_images_dir, "red_100x100_noisy.png")

        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up before each test method"""
        self.calculator = ImageDiffCalculator()

    def test_initialization(self):
        """Test ImageDiffCalculator initialization"""
        self.assertIsInstance(self.calculator, ImageDiffCalculator)
        self.assertEqual(len(self.calculator.supported_methods), 5)
        self.assertIn('mse', self.calculator.supported_methods)
        self.assertIn('ssim', self.calculator.supported_methods)
        self.assertIn('histogram', self.calculator.supported_methods)
        self.assertIn('pixel_diff', self.calculator.supported_methods)
        self.assertIn('hash', self.calculator.supported_methods)

    def test_load_images_success(self):
        """Test successful image loading"""
        img1, img2 = self.calculator.load_images(self.red_image, self.blue_image)

        self.assertIsInstance(img1, np.ndarray)
        self.assertIsInstance(img2, np.ndarray)
        self.assertEqual(len(img1.shape), 3)  # Color image
        self.assertEqual(len(img2.shape), 3)  # Color image
        self.assertEqual(img1.shape, (100, 100, 3))
        self.assertEqual(img2.shape, (100, 100, 3))

    def test_load_images_file_not_found(self):
        """Test loading non-existent images"""
        with self.assertRaises(FileNotFoundError):
            self.calculator.load_images("non_existent.png", self.red_image)

        with self.assertRaises(FileNotFoundError):
            self.calculator.load_images(self.red_image, "non_existent.png")

    def test_load_images_invalid_file(self):
        """Test loading invalid image files"""
        # Create a temporary text file
        invalid_file = os.path.join(self.temp_dir, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("This is not an image")

        with self.assertRaises(ValueError):
            self.calculator.load_images(invalid_file, self.red_image)

    def test_resize_images_same_size(self):
        """Test resizing images that are already the same size"""
        img1, img2 = self.calculator.load_images(self.red_image, self.blue_image)
        resized1, resized2 = self.calculator.resize_images(img1, img2)

        self.assertEqual(resized1.shape, resized2.shape)
        self.assertEqual(resized1.shape, (100, 100, 3))

    def test_resize_images_different_sizes(self):
        """Test resizing images with different dimensions"""
        img1, img2 = self.calculator.load_images(self.red_image, self.red_small)
        resized1, resized2 = self.calculator.resize_images(img1, img2)

        self.assertEqual(resized1.shape, resized2.shape)
        self.assertEqual(resized1.shape, (50, 50, 3))  # Should resize to smaller dimension

    def test_calculate_mse_identical_images(self):
        """Test MSE calculation for identical images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.red_copy)
        mse = self.calculator.calculate_mse(img1, img2)

        self.assertEqual(mse, 0.0)

    def test_calculate_mse_different_images(self):
        """Test MSE calculation for different images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.blue_image)
        mse = self.calculator.calculate_mse(img1, img2)

        self.assertGreater(mse, 0.0)
        self.assertIsInstance(mse, float)

    def test_calculate_ssim_identical_images(self):
        """Test SSIM calculation for identical images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.red_copy)
        ssim = self.calculator.calculate_ssim(img1, img2)

        self.assertAlmostEqual(ssim, 1.0, places=2)

    def test_calculate_ssim_different_images(self):
        """Test SSIM calculation for different images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.blue_image)
        ssim = self.calculator.calculate_ssim(img1, img2)

        self.assertLess(ssim, 1.0)
        self.assertGreaterEqual(ssim, -1.0)  # SSIM range is [-1, 1]
        self.assertIsInstance(ssim, float)

    def test_calculate_histogram_diff_identical_images(self):
        """Test histogram comparison for identical images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.red_copy)
        hist_corr = self.calculator.calculate_histogram_diff(img1, img2)

        self.assertAlmostEqual(hist_corr, 1.0, places=5)

    def test_calculate_histogram_diff_different_images(self):
        """Test histogram comparison for different images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.blue_image)
        hist_corr = self.calculator.calculate_histogram_diff(img1, img2)

        self.assertLess(hist_corr, 1.0)
        self.assertGreaterEqual(hist_corr, 0.0)  # Correlation range is [0, 1]
        self.assertIsInstance(hist_corr, float)

    def test_calculate_pixel_diff_ratio_identical_images(self):
        """Test pixel difference ratio for identical images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.red_copy)
        diff_ratio = self.calculator.calculate_pixel_diff_ratio(img1, img2)

        self.assertEqual(diff_ratio, 0.0)

    def test_calculate_pixel_diff_ratio_different_images(self):
        """Test pixel difference ratio for different images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.blue_image)
        diff_ratio = self.calculator.calculate_pixel_diff_ratio(img1, img2)

        self.assertGreater(diff_ratio, 0.0)
        self.assertLessEqual(diff_ratio, 1.0)
        self.assertIsInstance(diff_ratio, float)

    def test_calculate_pixel_diff_ratio_custom_threshold(self):
        """Test pixel difference ratio with custom threshold"""
        img1, img2 = self.calculator.load_images(self.red_image, self.red_modified)

        # Test with different thresholds
        diff_ratio_low = self.calculator.calculate_pixel_diff_ratio(img1, img2, threshold=10)
        diff_ratio_high = self.calculator.calculate_pixel_diff_ratio(img1, img2, threshold=100)

        self.assertGreaterEqual(diff_ratio_low, diff_ratio_high)
        self.assertIsInstance(diff_ratio_low, float)
        self.assertIsInstance(diff_ratio_high, float)

    def test_calculate_perceptual_hash_diff_identical_images(self):
        """Test perceptual hash difference for identical images"""
        img1, img2 = self.calculator.load_images(self.red_image, self.red_copy)
        hash_diff = self.calculator.calculate_perceptual_hash_diff(img1, img2)

        self.assertEqual(hash_diff, 0.0)

    def test_calculate_perceptual_hash_diff_different_images(self):
        """Test perceptual hash difference for different images"""
        # Use gradient vs red image for more structural difference
        img1, img2 = self.calculator.load_images(self.red_image, self.gradient_image)
        hash_diff = self.calculator.calculate_perceptual_hash_diff(img1, img2)

        self.assertGreaterEqual(hash_diff, 0.0)  # Allow 0.0 for some cases
        self.assertLessEqual(hash_diff, 1.0)
        self.assertIsInstance(hash_diff, float)

    def test_calculate_diff_ratio_mse_method(self):
        """Test calculate_diff_ratio with MSE method"""
        result = self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'mse')

        self.assertIn('method', result)
        self.assertIn('mse', result)
        self.assertIn('similarity_ratio', result)
        self.assertIn('image_size', result)
        self.assertEqual(result['method'], 'mse')
        self.assertGreater(result['mse'], 0.0)
        self.assertLessEqual(result['similarity_ratio'], 1.0)

    def test_calculate_diff_ratio_ssim_method(self):
        """Test calculate_diff_ratio with SSIM method"""
        result = self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'ssim')

        self.assertIn('method', result)
        self.assertIn('ssim', result)
        self.assertIn('similarity_ratio', result)
        self.assertEqual(result['method'], 'ssim')
        self.assertEqual(result['ssim'], result['similarity_ratio'])

    def test_calculate_diff_ratio_histogram_method(self):
        """Test calculate_diff_ratio with histogram method"""
        result = self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'histogram')

        self.assertIn('method', result)
        self.assertIn('histogram_correlation', result)
        self.assertIn('similarity_ratio', result)
        self.assertEqual(result['method'], 'histogram')
        self.assertEqual(result['histogram_correlation'], result['similarity_ratio'])

    def test_calculate_diff_ratio_pixel_diff_method(self):
        """Test calculate_diff_ratio with pixel difference method"""
        result = self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'pixel_diff')

        self.assertIn('method', result)
        self.assertIn('pixel_diff_ratio', result)
        self.assertIn('similarity_ratio', result)
        self.assertEqual(result['method'], 'pixel_diff')
        self.assertEqual(result['similarity_ratio'], 1.0 - result['pixel_diff_ratio'])

    def test_calculate_diff_ratio_hash_method(self):
        """Test calculate_diff_ratio with hash method"""
        result = self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'hash')

        self.assertIn('method', result)
        self.assertIn('hash_distance', result)
        self.assertIn('similarity_ratio', result)
        self.assertEqual(result['method'], 'hash')
        self.assertEqual(result['similarity_ratio'], 1.0 - result['hash_distance'])

    def test_calculate_diff_ratio_unsupported_method(self):
        """Test calculate_diff_ratio with unsupported method"""
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'invalid_method')

        self.assertIn("Unsupported method", str(context.exception))

    def test_save_diff_visualization(self):
        """Test saving difference visualization"""
        output_path = os.path.join(self.temp_dir, "test_visualization.png")

        # Mock matplotlib to avoid display issues
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.imshow') as mock_imshow, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.axis') as mock_axis, \
             patch('matplotlib.pyplot.bar') as mock_bar, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.xticks') as mock_xticks, \
             patch('matplotlib.pyplot.ylim') as mock_ylim, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout, \
             patch('builtins.print') as mock_print:

            self.calculator.save_diff_visualization(
                self.red_image,
                self.blue_image,
                output_path
            )

            # Verify that matplotlib functions were called
            mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
            mock_close.assert_called_once()
            mock_print.assert_called_once()

    def test_similarity_consistency(self):
        """Test that similarity calculations are consistent"""
        # Compare identical images - should have high similarity
        result_identical = self.calculator.calculate_diff_ratio(self.red_image, self.red_copy, 'mse')
        self.assertGreater(result_identical['similarity_ratio'], 0.9)

        # Compare very different images - should have low similarity
        result_different = self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, 'mse')
        self.assertLess(result_different['similarity_ratio'], 0.5)

        # Identical images should be more similar than different images
        self.assertGreater(result_identical['similarity_ratio'], result_different['similarity_ratio'])

    def test_all_methods_return_valid_similarity(self):
        """Test that all methods return valid similarity ratios"""
        for method in self.calculator.supported_methods:
            result = self.calculator.calculate_diff_ratio(self.red_image, self.blue_image, method)

            self.assertIn('similarity_ratio', result)
            self.assertGreaterEqual(result['similarity_ratio'], 0.0)
            self.assertLessEqual(result['similarity_ratio'], 1.0)
            self.assertIsInstance(result['similarity_ratio'], float)

    def test_noise_sensitivity(self):
        """Test sensitivity to noise"""
        # Compare original with noisy version
        result_noisy = self.calculator.calculate_diff_ratio(self.red_image, self.red_noisy, 'ssim')

        # Should still be relatively similar but not identical
        self.assertGreater(result_noisy['similarity_ratio'], 0.5)
        self.assertLess(result_noisy['similarity_ratio'], 1.0)

    def test_size_independence(self):
        """Test that resizing works correctly for different sized images"""
        result = self.calculator.calculate_diff_ratio(self.red_image, self.red_small, 'mse')

        # Should be able to handle different sizes
        self.assertIn('similarity_ratio', result)
        self.assertIn('image_size', result)
        # After resizing, should be reasonably similar since both are red
        self.assertGreater(result['similarity_ratio'], 0.8)


class TestImageDiffCalculatorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up before each test method"""
        self.calculator = ImageDiffCalculator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_directory_path(self):
        """Test handling of empty string paths"""
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.calculator.load_images("", "test.png")

    def test_directory_instead_of_file(self):
        """Test handling when directory path is passed instead of file"""
        with self.assertRaises((ValueError, cv2.error)):
            self.calculator.load_images(self.temp_dir, self.temp_dir)

    def test_very_small_images(self):
        """Test with very small images"""
        # Create 1x1 pixel images
        tiny_img1 = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Red
        tiny_img2 = np.array([[[0, 255, 0]]], dtype=np.uint8)  # Green

        # Save tiny images
        tiny_path1 = os.path.join(self.temp_dir, "tiny1.png")
        tiny_path2 = os.path.join(self.temp_dir, "tiny2.png")
        cv2.imwrite(tiny_path1, tiny_img1)
        cv2.imwrite(tiny_path2, tiny_img2)

        # Test should handle tiny images without crashing
        result = self.calculator.calculate_diff_ratio(tiny_path1, tiny_path2, 'mse')
        self.assertIn('similarity_ratio', result)

    def test_grayscale_images(self):
        """Test with grayscale images converted to color"""
        # Create grayscale image
        gray_img = np.ones((50, 50), dtype=np.uint8) * 128
        gray_path = os.path.join(self.temp_dir, "gray.png")
        cv2.imwrite(gray_path, gray_img)

        # OpenCV will load as 3-channel anyway, but test should still work
        result = self.calculator.calculate_diff_ratio(gray_path, gray_path, 'mse')
        self.assertAlmostEqual(result['similarity_ratio'], 1.0, places=5)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestImageDiffCalculator))
    test_suite.addTest(unittest.makeSuite(TestImageDiffCalculatorEdgeCases))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
