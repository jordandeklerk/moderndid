"""Measure complete estimator calls over named workloads."""

from benchmarks.common import EstimatorBenchmark


class ATTgt(EstimatorBenchmark):
    """Measure staggered difference-in-differences estimation."""

    estimator = "att_gt"
    params = ("small", "baseline", "rows", "periods", "cohorts", "covariates", "bootstrap", "ipw", "reg")
    param_names = ("workload",)

    def time_estimate(self, profile):
        """Measure one complete estimator call."""
        self.estimate()


class TripleDifferences(EstimatorBenchmark):
    """Measure staggered triple-differences estimation."""

    estimator = "ddd"
    params = ("small", "baseline", "rows", "periods", "cohorts", "covariates", "bootstrap", "ipw", "reg")
    param_names = ("workload",)

    def time_estimate(self, profile):
        """Measure one complete estimator call."""
        self.estimate()


class ContinuousTreatment(EstimatorBenchmark):
    """Measure continuous-treatment estimation and inference."""

    estimator = "cont_did"
    params = ("small", "baseline", "rows", "periods", "cohorts", "bootstrap")
    param_names = ("workload",)

    def time_estimate(self, profile):
        """Measure one complete estimator call."""
        self.estimate()


class IntertemporalTreatment(EstimatorBenchmark):
    """Measure intertemporal treatment-effect estimation."""

    estimator = "did_multiplegt"
    params = ("small", "baseline", "rows", "periods", "cohorts", "covariates", "bootstrap", "normalized")
    param_names = ("workload",)

    def time_estimate(self, profile):
        """Measure one complete estimator call."""
        self.estimate()


class DoublyRobustDiD(EstimatorBenchmark):
    """Measure two-period doubly robust difference-in-differences estimation."""

    estimator = "drdid"
    params = ("small", "baseline", "rows", "covariates")
    param_names = ("workload",)

    def time_estimate(self, profile):
        """Measure one complete estimator call."""
        self.estimate()


class InverseProbabilityWeightingDiD(EstimatorBenchmark):
    """Measure two-period inverse probability weighted difference-in-differences estimation."""

    estimator = "ipwdid"
    params = ("small", "baseline", "rows", "covariates")
    param_names = ("workload",)

    def time_estimate(self, profile):
        """Measure one complete estimator call."""
        self.estimate()


class OutcomeRegressionDiD(EstimatorBenchmark):
    """Measure two-period outcome regression difference-in-differences estimation."""

    estimator = "ordid"
    params = ("small", "baseline", "rows", "covariates")
    param_names = ("workload",)

    def time_estimate(self, profile):
        """Measure one complete estimator call."""
        self.estimate()
