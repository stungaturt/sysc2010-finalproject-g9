import unittest
import numpy as np
import os
import data_loader
from functions import apply_butterworth_filter, get_basic_statistics

class TestSignalSystem(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy CSV for testing
        self.test_file = "test_data.csv"
        t = np.linspace(0, 1, 100)
        s = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(100)
        data = np.column_stack((t, s))
        np.savetxt(self.test_file, data, delimiter=",", header="time,signal", comments='')

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_data_loading(self):
        """Verify CSV loading and shape [cite: 200]"""
        time, signal = load_sensor_data(self.test_file)
        self.assertIsNotNone(time)
        self.assertEqual(len(time), 100)