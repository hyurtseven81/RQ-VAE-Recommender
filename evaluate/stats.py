from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    cohens_d: float


def paired_bootstrap_test(
    user_metrics_a: dict[int, float],
    user_metrics_b: dict[int, float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap test on per-user metric differences.

    Both dicts must have the same user IDs. Computes:
    - mean_diff = mean(a - b)
    - 95% CI via bootstrap percentile method
    - two-sided p-value: proportion of bootstrap samples where diff <= 0
      (if mean_diff > 0) or >= 0 (if mean_diff < 0).  When mean_diff == 0
      p_value is set to 1.0.
    - Cohen's d on the per-user differences
    """
    if set(user_metrics_a.keys()) != set(user_metrics_b.keys()):
        raise ValueError("user_metrics_a and user_metrics_b must have the same user IDs")

    user_ids = sorted(user_metrics_a.keys())
    a = np.array([user_metrics_a[u] for u in user_ids], dtype=float)
    b = np.array([user_metrics_b[u] for u in user_ids], dtype=float)
    diffs = a - b

    mean_diff = float(np.mean(diffs))

    # Bootstrap resampling
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = float(np.mean(sample))

    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # Two-sided p-value
    if mean_diff > 0:
        p_value = float(np.mean(boot_means <= 0))
    elif mean_diff < 0:
        p_value = float(np.mean(boot_means >= 0))
    else:
        p_value = 1.0

    # Cohen's d: mean(diffs) / std(diffs)  (population std of the diffs)
    std_diffs = float(np.std(diffs, ddof=1))
    if std_diffs == 0.0:
        cohens_d = 0.0
    else:
        cohens_d = mean_diff / std_diffs

    return BootstrapResult(
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        cohens_d=cohens_d,
    )


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Returns significance flags after Holm-Bonferroni correction.

    Steps:
    1. Sort p-values in ascending order, keeping track of original indices.
    2. For rank i (1-indexed), the adjusted threshold is alpha / (m - i + 1)
       where m is the total number of tests.
    3. A hypothesis is rejected if ALL hypotheses with smaller p-values are
       also rejected AND p_i <= adjusted threshold.
    4. Once the first non-rejection occurs, all subsequent hypotheses (in
       sorted order) are also not rejected.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * m
    stop = False
    for rank, (original_idx, p) in enumerate(indexed, start=1):
        if stop:
            break
        threshold = alpha / (m - rank + 1)
        if p <= threshold:
            significant[original_idx] = True
        else:
            stop = True

    return significant
