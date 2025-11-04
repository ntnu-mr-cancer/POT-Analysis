from sklearn.utils import resample
from math import sqrt
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    DeLong test for two correlated ROC AUCs (same subjects).
    Returns: two-sided p-value (float).
    """
    y = np.asarray(ground_truth, dtype=int)
    s1 = np.asarray(predictions_one, dtype=float)
    s2 = np.asarray(predictions_two, dtype=float)

    if y.ndim != 1 or s1.ndim != 1 or s2.ndim != 1:
        raise ValueError("All inputs must be 1-D arrays.")
    if not (len(y) == len(s1) == len(s2)):
        raise ValueError("Arrays must have the same length.")
    if y.min() == y.max():
        raise ValueError("ground_truth must contain both classes 0 and 1.")

    def _midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            # midrank over the tie block [i, j)
            T[i:j] = 0.5 * (i + j - 1) + 1.0
            i = j
        out = np.empty(N, dtype=float)
        out[J] = T
        return out

    # sort so positives (1) come first, as required by DeLong derivation
    order = np.argsort(-y)
    y_sorted = y[order]
    m = int(y_sorted.sum())
    n = len(y_sorted) - m

    preds_sorted_T = np.vstack([s1[order], s2[order]])  # shape (2, m+n)
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]

    # midranks
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])     # (2, m)
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])     # (2, n)
    tz = np.vstack([_midrank(preds_sorted_T[r]) for r in range(2)])  # (2, m+n)

    # AUCs via rank formula
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)

    # DeLong covariance components
    v01 = (tz[:, :m] - tx) / n              # (2, m)
    v10 = 1.0 - (tz[:, m:] - ty) / m        # (2, n)

    s01 = np.cov(v01, bias=False)           # (2, 2)
    s10 = np.cov(v10, bias=False)           # (2, 2)
    cov = s01 / m + s10 / n                 # (2, 2)

    # variance of AUC difference (model1 - model2)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0:
        var = np.finfo(float).eps  # numerical guard for tiny/identical cases

    z = diff / np.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(p)


def calculate_confidence_interval(proportion, total_cases, z=1.96):
    """
    Calculate the confidence interval for a proportion at a given confidence level.

    Parameters:
        proportion (float): The proportion as a percentage (0-100). Example: 25 for 25%.
        total_cases (int): The total number of cases used to calculate the proportion.
        z (float): The Z-score for the desired confidence level. Default is 1.96 (for 95% confidence).

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval as percentages (0-100).

    Formula:
        CI = p ± Z * sqrt((p * (1 - p)) / n)
        Where:
            p = proportion (in decimal form, e.g., 0.25 for 25%)
            n = total number of cases
            Z = Z-score corresponding to the confidence level (default 1.96 for 95%)

    Notes:
        - If `total_cases` is 0, the function returns (0, 0) to handle division by zero.
        - The bounds are clamped to 0-100 to ensure valid percentage values.
    """
    if total_cases == 0:
        return (0, 0)  # Avoid division by zero

    # Convert proportion from percentage to decimal
    proportion = proportion / 100

    # Calculate the margin of error
    margin_of_error = z * sqrt((proportion * (1 - proportion)) / total_cases)

    # Calculate lower and upper bounds, and convert back to percentage
    lower_bound = max(0, (proportion - margin_of_error) * 100)
    upper_bound = min(100, (proportion + margin_of_error) * 100)

    return lower_bound, upper_bound


def bootstrap_auc_ci(ground_truth, scores, n_bootstraps=1000, ci=95):
    """
    Calculate the 95% confidence interval for AUC using bootstrapping.

    Parameters:
        ground_truth (array-like): Ground truth binary labels.
        scores (array-like): Predicted scores or probabilities.
        n_bootstraps (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage (default is 95).

    Returns:
        tuple: Lower and upper bounds of the AUC confidence interval.
    """
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(ground_truth)), replace=True)
        if len(set(ground_truth[indices])) < 2:
            continue  # Skip if resampling creates a single-class case
        fpr, tpr, _ = roc_curve(ground_truth[indices], scores[indices])
        bootstrapped_scores.append(auc(fpr, tpr))

    # Compute CI percentiles
    lower_bound = np.percentile(bootstrapped_scores, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_scores, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound


def bootstrap_ap_ci(ground_truth, scores, n_bootstraps=1000, ci=95):
    """
    Calculate the 95% confidence interval for Average Precision (AP) using bootstrapping.

    Parameters:
        ground_truth (array-like): Ground truth binary labels.
        scores (array-like): Predicted scores or probabilities.
        n_bootstraps (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage (default is 95%).

    Returns:
        tuple: Lower and upper bounds of the AP confidence interval.
    """
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(ground_truth)), replace=True)
        if len(set(ground_truth[indices])) < 2:
            continue  # Skip if resampling creates a single-class case
        ap = average_precision_score(ground_truth[indices], scores[indices])
        bootstrapped_scores.append(ap)

    # Compute CI percentiles
    lower_bound = np.percentile(bootstrapped_scores, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_scores, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound


def perform_mcnemar_test_specificity_accuracy(ground_truth, ai_predictions, radiologist_predictions, pirads_threshold):
    """
    Perform McNemar tests comparing AI predictions with radiologist predictions at a specific PIRADS threshold.

    Args:
        ground_truth (array-like): Ground truth labels (0/1)
        ai_predictions (array-like): AI binary predictions (0/1)
        radiologist_predictions (array-like): Radiologist binary predictions (0/1)
        pirads_threshold (int): PIRADS threshold used (3 or 4) - for logging purposes

    Returns:
        dict: Dictionary containing p-values for specificity and accuracy comparisons
    """
    def contingency_from_bool(a_bool, b_bool):
        both_true = np.sum(np.logical_and(a_bool, b_bool))
        a_true_b_false = np.sum(np.logical_and(a_bool, np.logical_not(b_bool)))
        a_false_b_true = np.sum(np.logical_and(np.logical_not(a_bool), b_bool))
        both_false = np.sum(np.logical_and(
            np.logical_not(a_bool), np.logical_not(b_bool)))
        return np.array([[both_true, a_true_b_false], [a_false_b_true, both_false]])

    results = {}

    # Specificity calculation (for negative cases only)
    neg_idx = np.where(np.array(ground_truth) == 0)[0]
    if len(neg_idx) > 0:
        ai_neg = (ai_predictions[neg_idx] == 0)
        rad_neg = (radiologist_predictions[neg_idx] == 0)

        table_specificity = contingency_from_bool(ai_neg, rad_neg)

        # Calculate discordant pairs
        false_specificity = int(table_specificity[0, 1])
        true_specificity = int(table_specificity[1, 0])
        use_exact = (false_specificity + true_specificity) < 25 or min(
            false_specificity, true_specificity) < 10

        mcnemar_result = mcnemar(
            table_specificity,
            exact=use_exact,
            correction=not use_exact
        )
        results[f'specificity_vs_pirads{pirads_threshold}_p'] = mcnemar_result.pvalue

    # Accuracy calculation
    table_accuracy = contingency_from_bool(
        ai_predictions == ground_truth,
        radiologist_predictions == ground_truth
    )

    # Calculate discordant pairs
    false_accuracy = int(table_accuracy[0, 1])
    true_accuracy = int(table_accuracy[1, 0])
    use_exact = (false_accuracy +
                 true_accuracy) < 25 or min(false_accuracy, true_accuracy) < 10

    mcnemar_result = mcnemar(
        table_accuracy,
        exact=use_exact,
        correction=not use_exact
    )
    results[f'accuracy_vs_pirads{pirads_threshold}_p'] = mcnemar_result.pvalue

    return results
