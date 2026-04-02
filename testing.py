import unittest
import numpy as np
import os
from data_loader import load_csv 
from functions import remove_baseline, lowpass, highpass, bandpass, features

class TestSignalSystem(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "temp_signal.csv"
        t = np.linspace(0, 1, 100)
        s = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(100)
        data = np.column_stack((t, s))
        np.savetxt(self.test_file, data, delimiter=",")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_data_loading(self):
        """Verify CSV loading and shape"""
        time, signal = load_csv(self.test_file)
        self.assertIsNotNone(time)
        self.assertEqual(len(time), 100)

    def test_statistics(self):
        """Verify statistical calculations"""
        data = np.array([1, 2, 3, 4, 5])
        stats = features(data)
        self.assertEqual(stats['mean'], 3.0)
        self.assertIn('rms', stats)
        self.assertAlmostEqual(stats['rms'], 3.3166, places=3)

    def test_all_filters(self):
        """Verify that all three filter functions return correct shapes"""
        _, signal = load_csv(self.test_file) 
        fs = 100
        filter_functions = [lowpass, highpass, bandpass]
    
        for func in filter_functions:
            filtered = func(signal, fs)
            self.assertEqual(len(filtered), len(signal), f"{func.__name__} failed")

    def test_robustness_empty_file(self):
        """Ensure system handles empty input"""
        empty_file = "empty.csv"
        with open(empty_file, 'w') as f:
            f.write("")
        
        # Call the function once and check the result
        result = load_csv(empty_file)
        self.assertIsNone(result[0]) # Check if 'time' is None
        
        # Test 2: Check if functions raise ValueError for empty lists
        with self.assertRaises(ValueError):
            remove_baseline([])

if __name__ == "__main__":
    unittest.main(verbosity=2)