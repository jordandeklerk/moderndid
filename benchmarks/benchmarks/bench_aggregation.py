"""Measure aggregation after fitting group-time effects."""

from benchmarks.common import make_aggregation


class Aggregation:
    """Measure aggregation from a fitted result."""

    params = (("att_gt", "ddd"), ("small", "baseline", "rows"), ("simple", "group", "dynamic"))
    param_names = ("estimator", "workload", "aggregation")
    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    version = "1"
    timeout = 120

    def setup(self, estimator, profile, aggregation):
        """Fit group-time effects and warm aggregation."""
        self.aggregate = make_aggregation(estimator, profile, aggregation)

    def time_aggregate(self, estimator, profile, aggregation):
        """Measure one aggregation call."""
        self.aggregate()
