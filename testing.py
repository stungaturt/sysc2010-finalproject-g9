import unittest
import numpy as np
import os
import data_loader
from functions import apply_butterworth_filter, get_basic_statistics

class TestSignalSystem(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "test_data.csv"
        t = np.linspace(0, 1, 100)
        s = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(100)
        data = np.column_stack((t, s))
        np.savetxt(self.test_file, data, delimiter=",", header="time,signal", comments='')

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_data_loading(self):
        """Verify CSV loading and shape"""
        time, signal = load_sensor_data(self.test_file) # edit function name
        self.assertIsNotNone(time)
        self.assertEqual(len(time), 100)

    def test_statistics(self):
        """Verify statistical calculations"""
        data = np.array([1, 2, 3, 4, 5])
        stats = get_basic_statistics(data)
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['range'], 4.0)

    def test_filtering_output_shape(self):
        """Ensure filtered signal length matches input"""
        _, signal = load_sensor_data(self.test_file)
        filtered = apply_butterworth_filter(signal, cutoff=10, fs=100)
        self.assertEqual(len(filtered), len(signal))

    def test_robustness_empty_file(self):
        """Ensure system doesn't crash on empty input"""
        empty_file = "empty.csv"
        with open(empty_file, 'w') as f:
            f.write("")
        time, signal = load_sensor_data(empty_file)
        self.assertIsNone(time)
        os.remove(empty_file)

if __name__ == "__main__":
    unittest.main()