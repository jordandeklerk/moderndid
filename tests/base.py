"""Base test class for drdidsynth tests."""

import unittest

import numpy as np
import pandas as pd


class BaseTest(unittest.TestCase):
    """Base test class for all drdidsynth tests."""

    @classmethod
    def setUpClass(cls):
        """Set up resources for the test class."""
        super().setUpClass()

    def setUp(self):
        """Set up resources for each test method."""
        super().setUp()
        np.random.seed(42)

    # Method intentionally left as a hook for subclasses
    # pylint: disable=useless-parent-delegation
    def tearDown(self):
        """Clean up resources after each test method."""
        super().tearDown()

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after the test class."""
        super().tearDownClass()

    @staticmethod
    def create_test_data(n_units=100, n_time=5, treatment_time=3):
        """Create synthetic data for testing.

        Parameters
        ----------
        n_units, default=100
            Number of units (e.g., individuals, firms, countries).
        n_time, default=5
            Number of time periods.
        treatment_time, default=3
            Time period when treatment starts (0-indexed).

        Returns
        -------
        dict
            Dictionary containing:
            - 'df': pandas DataFrame with panel data
            - 'treatment_effects': numpy array of true treatment effects
        """
        unit_ids = np.arange(n_units)
        time_ids = np.arange(n_time)

        treatment = np.zeros(n_units, dtype=bool)
        treatment[: n_units // 2] = True

        features = np.random.normal(size=(n_units, 2))

        unit_effects = 0.5 * features[:, 0] + np.random.normal(size=n_units, scale=0.5)
        time_effects = 0.3 * np.arange(n_time) + np.random.normal(size=n_time, scale=0.2)
        true_effects = 1.0 + 0.5 * features[:, 1]

        outcomes = np.zeros((n_units, n_time))
        for i in range(n_units):
            for t in range(n_time):
                outcomes[i, t] = unit_effects[i] + time_effects[t]

                if treatment[i] and t >= treatment_time:
                    outcomes[i, t] += true_effects[i]

                outcomes[i, t] += np.random.normal(scale=0.2)

        data = []
        for i in unit_ids:
            for t in time_ids:
                data.append(
                    {
                        "unit_id": i,
                        "time_id": t,
                        "treatment": treatment[i] and t >= treatment_time,
                        "X1": features[i, 0],
                        "X2": features[i, 1],
                        "outcome": outcomes[i, t],
                    }
                )

        df = pd.DataFrame(data)

        return {"df": df, "treatment_effects": true_effects[treatment]}

    @staticmethod
    def assert_close(actual, expected, rtol=1e-5, atol=1e-8, msg=None):
        """Assert that arrays are close within a tolerance.

        Parameters
        ----------
        actual
            Actual values.
        expected
            Expected values.
        rtol, default=1e-5
            Relative tolerance.
        atol, default=1e-8
            Absolute tolerance.
        msg, default=None
            Optional error message.
        """
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)

    @staticmethod
    def assert_frame_equal(
        actual,
        expected,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_names=True,
        msg=None,
    ):
        """Assert that pandas DataFrames are equal.

        Parameters
        ----------
        actual
            Actual DataFrame.
        expected
            Expected DataFrame.
        check_dtype, default=True
            Whether to check data types.
        check_index_type, default=True
            Whether to check index types.
        check_column_type, default=True
            Whether to check column types.
        check_frame_type, default=True
            Whether to check the DataFrame types.
        check_names, default=True
            Whether to check the index and column names.
        msg, default=None
            Optional error message.
        """
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=check_dtype,
            check_index_type=check_index_type,
            check_column_type=check_column_type,
            check_frame_type=check_frame_type,
            check_names=check_names,
            obj=msg,
        )
