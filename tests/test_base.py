"""Test the base test class functionality."""

import numpy as np
import pandas as pd

from .base import BaseTest


class TestBaseTest(BaseTest):
    """Test the functionality of the BaseTest class."""

    def test_create_test_data(self):
        data = self.create_test_data(n_units=10, n_time=4, treatment_time=2)

        self.assertIn("df", data)
        self.assertIn("treatment_effects", data)

        df = data["df"]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10 * 4)

        expected_columns = {"unit_id", "time_id", "treatment", "X1", "X2", "outcome"}
        self.assertEqual(set(df.columns), expected_columns)

        treated_units = df[df["treatment"]]["unit_id"].unique()
        self.assertTrue(all(unit_id < 5 for unit_id in treated_units))

        treated_times = df[df["treatment"]]["time_id"].unique()
        self.assertTrue(all(time_id >= 2 for time_id in treated_times))
        self.assertEqual(len(data["treatment_effects"]), 5)  # Half of units are treated

    def test_assert_close(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0000001, 2.0000002, 3.0000003])
        self.assert_close(a, b)

        c = np.array([1.1, 2.2, 3.3])
        with self.assertRaises(AssertionError):
            self.assert_close(a, c)

    def test_assert_frame_equal(self):
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.assert_frame_equal(df1, df2)

        df3 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})
        with self.assertRaises(AssertionError):
            self.assert_frame_equal(df1, df3)
