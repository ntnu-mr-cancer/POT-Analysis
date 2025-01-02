from sklearn.utils import resample
from math import sqrt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
from scipy.stats import norm


def compute_midrank(x):
    """Computes midranks."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    return T


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast implementation of DeLong's method for computing AUC variances."""
    m = label_1_count  # Number of positive samples
    # Number of negative samples
    n = predictions_sorted_transposed.shape[1] - m

    # Extract the positive and negative examples for both sets of predictions
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    # Calculate midranks for positive and negative examples
    tx = np.zeros(m)
    ty = np.zeros(n)
    tz = np.zeros(m + n)

    for r in range(predictions_sorted_transposed.shape[0]):
        tx += compute_midrank(positive_examples[r, :])
        ty += compute_midrank(negative_examples[r, :])
        tz += compute_midrank(predictions_sorted_transposed[r, :])

    # Normalize midranks
    tx /= m
    ty /= n
    tz /= (m + n)

    # The lengths of tx and ty should be consistent with tz
    aucs = tz - (np.concatenate([tx, ty]) / 2)
    auc_cov = aucs.var(ddof=1)

    return auc_cov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """Performs DeLong's test for comparing two ROC AUCs."""
    auc_one = roc_auc_score(ground_truth, predictions_one)
    auc_two = roc_auc_score(ground_truth, predictions_two)

    # Convert to numpy arrays for NumPy-based indexing
    ground_truth_np = np.array(ground_truth)
    predictions_one_np = np.array(predictions_one)
    predictions_two_np = np.array(predictions_two)

    # Sort predictions_one and use the order for both predictions
    order = np.argsort(-predictions_one_np)
    predictions_sorted = np.vstack(
        (predictions_one_np[order], predictions_two_np[order]))
    ground_truth_sorted = ground_truth_np[order]

    # Count of positive labels after sorting
    label_1_count = np.sum(ground_truth_sorted)

    # Ensure that label_1_count is consistent between predictions
    if predictions_sorted.shape[1] != len(ground_truth_sorted):
        raise ValueError(
            "Mismatch between number of samples in predictions and ground truth")

    auc_cov = fastDeLong(predictions_sorted, label_1_count)

    z = (auc_one - auc_two) / np.sqrt(np.abs(auc_cov))
    p_value = 2 * (1 - norm.cdf(np.abs(z)))
    return p_value


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
