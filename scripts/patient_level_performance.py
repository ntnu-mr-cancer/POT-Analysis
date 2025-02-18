"""
Patient-level performance analysis code, part of the
Analysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import delong_roc_test, calculate_confidence_interval, bootstrap_auc_ci
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, f1_score
from datetime import datetime


def evaluate_patient_level_performance(original_findings_statistics, log_file="patient_level_performance.log", output_dir='plots', alpha=0.5):
    """
    Evaluates AI and Radiologist performance in predicting significant prostate cancer (GGG > 1),
    including combined approaches ("AI or Radiologist" and "AI and Radiologist").

    Parameters:
        original_findings_statistics (dict): Contains results of various findings analyses from original data pre adjustment.
        log_file (str): Path to the log file for recording evaluation results.
        output_dir (str): Directory for saving plots.
        alpha (float): Weighting factor for combined AI and Radiologist scores (0 <= alpha <= 1).

    Returns:
        roc_data (dict): ROC curve data and metrics for AI, Radiologist, and combined approaches.
        operating_points (dict): Thresholds for AI and combined approaches at specific targets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract ground truth and predictions
    ground_truth = original_findings_statistics["ggg_stats"]["patient_has_overall_cspca"]
    highest_pirads = np.array(
        original_findings_statistics["radiologist_stats"]["highest_PIRADS_Score"])
    highest_ai_score = original_findings_statistics["ai_stats"]["highest_scores"]

    # Initialize dictionary for storing ROC and metrics data
    roc_data = {}

    # ROC curve and AUC for AI model (only AUC is stored here)
    fpr_ai, tpr_ai, thresholds_ai = roc_curve(ground_truth, highest_ai_score)
    roc_auc_ai = auc(fpr_ai, tpr_ai)
    roc_data["AI"] = {'AUC': roc_auc_ai}

    # ROC curve and AUC for Radiologists (with full metrics)
    fpr_radiologist, tpr_radiologist, _ = roc_curve(
        ground_truth, highest_pirads)
    roc_auc_radiologist = auc(fpr_radiologist, tpr_radiologist)

    # Radiologist operating point with threshold 3
    radiologist_binary_predictions_3 = (highest_pirads >= 3).astype(int)

    def calculate_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'Sensitivity (TPR)': tp / (tp + fn),
            'Specificity (TNR)': tn / (tn + fp),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
            'True Positives Percentage': tp/len(y_true),
            'False Positives Percentage': fp/len(y_true),
            'True Negatives Percentage': tn/len(y_true),
            'False Negatives Percentage': fn/len(y_true),
            'Positives Count': tp + fp,
            'Negatives Count': tn + fn,
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn
        }

    metrics_radiologist_3 = calculate_metrics(
        ground_truth, radiologist_binary_predictions_3)
    metrics_radiologist_3['AUC'] = roc_auc_radiologist
    roc_data["Radiologist @ PIRADS ≥ 3"] = metrics_radiologist_3

    # Radiologist operating point with threshold 4
    radiologist_binary_predictions_4 = (highest_pirads >= 4).astype(int)
    metrics_radiologist_4 = calculate_metrics(
        ground_truth, radiologist_binary_predictions_4)
    metrics_radiologist_4['AUC'] = roc_auc_radiologist
    roc_data["Radiologist @ PIRADS ≥ 4"] = metrics_radiologist_4

    # Perform DeLong's test between AI and Radiologists' ROC curves
    p_value = delong_roc_test(ground_truth, highest_ai_score, highest_pirads)

    def find_threshold(tpr_target=None, fpr_target=None, sensitivity_target=None):
        idx = None

        if sensitivity_target is not None:
            # Find the index where the sensitivity (tpr) is closest to the target
            idxs = np.where(tpr_ai >= sensitivity_target)[0]
            if len(idxs) > 0:
                idx = idxs[0]
                return thresholds_ai[idx] if idx != -1 else None
            else:
                return None

        elif tpr_target is not None:
            idx = np.abs(tpr_ai - tpr_target).argmin()
            return thresholds_ai[idx] if idx != -1 else None

        elif fpr_target is not None:
            idx = np.abs(fpr_ai - fpr_target).argmin()
            return thresholds_ai[idx] if idx != -1 else None

    operating_points = {
        'Study thd': 0.73,
        'Optimized thd': find_threshold(tpr_target=metrics_radiologist_3['Sensitivity (TPR)'])
    }

    # Calculate and store metrics for each AI operating point in roc_data
    roc_data["AI"]["Operating Points"] = {}
    for op_name, threshold in operating_points.items():
        if threshold is not None:
            ai_predictions = (highest_ai_score >= threshold).astype(int)
            metrics = calculate_metrics(ground_truth, ai_predictions)
            metrics['Threshold'] = threshold
            roc_data["AI"]["Operating Points"][(op_name, threshold)] = metrics

    # Log metrics for Both_AI_and_Radiologist_Score at specific operating points
    roc_data["Both_AI_and_Radiologist_Score"] = {}
    for op_name, threshold in operating_points.items():
        if threshold is not None:
            # Both: AI >= threshold and Radiologist > 3
            and_predictions = ((highest_ai_score >= threshold) & (
                highest_pirads > 3)).astype(int)

            # Calculate metrics for Both criteria
            fpr_and, tpr_and, _ = roc_curve(ground_truth, and_predictions)
            roc_auc_and = auc(fpr_and, tpr_and)

            # Calculate full metrics and include AUC
            metrics_and = calculate_metrics(ground_truth, and_predictions)
            metrics_and['AUC'] = roc_auc_and
            roc_data["Both_AI_and_Radiologist_Score"][(
                op_name, threshold)] = metrics_and

    # Log metrics for Either_AI_or_Radiologist_Score at specific operating points
    roc_data["Either_AI_or_Radiologist_Score"] = {}
    for op_name, threshold in operating_points.items():
        if threshold is not None:
            # Either: AI >= threshold or Radiologist > 3
            or_predictions = ((highest_ai_score >= threshold) | (
                highest_pirads > 3)).astype(int)

            # Calculate metrics for Either criteria
            fpr_or, tpr_or, _ = roc_curve(ground_truth, or_predictions)
            roc_auc_or = auc(fpr_or, tpr_or)

            # Calculate full metrics and include AUC
            metrics_or = calculate_metrics(ground_truth, or_predictions)
            metrics_or['AUC'] = roc_auc_or
            roc_data["Either_AI_or_Radiologist_Score"][(
                op_name, threshold)] = metrics_or

    # Main ROC plot for AI, Radiologist, and Combined Methods
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 14})

    # Plot AI ROC curve with operating points
    plt.plot(fpr_ai, tpr_ai,
             label=f'AI (continuous) AUC = {roc_auc_ai * 100:.2f}%', color='b', linewidth=2)
    for op_name, threshold in operating_points.items():
        if threshold is not None:
            plt.scatter((1-roc_data["AI"]["Operating Points"][(op_name, threshold)]["Specificity (TNR)"]),
                        roc_data["AI"]["Operating Points"][(
                            op_name, threshold)]["Sensitivity (TPR)"],
                        label=f'AI @ {threshold} - {op_name}', s=100)

    # Plot Radiologist ROC and operating points
    plt.plot(fpr_radiologist, tpr_radiologist,
             label=f'Radiologist AUC = {roc_auc_radiologist * 100:.2f}%', color='r', linestyle='--', linewidth=2)

    # Radiologist @ PIRADS ≥ 3
    plt.scatter(1 - metrics_radiologist_3["Specificity (TNR)"], metrics_radiologist_3["Sensitivity (TPR)"],
                label='Radiologist @ PIRADS ≥ 3', color='r', marker='^', s=100)

    # Radiologist @ PIRADS ≥ 4
    plt.scatter(1 - metrics_radiologist_4["Specificity (TNR)"], metrics_radiologist_4["Sensitivity (TPR)"],
                label='Radiologist @ PIRADS ≥ 4', color='g', marker='^', s=100)

    # Plot AI AND Radiologist ROC and operating points
    plt.plot(fpr_and, tpr_and,
             label=f'Radiologist AND AI AUC = {roc_auc_and * 100:.2f}%', color='y', linestyle='-.', linewidth=2)

    # AND Radiologist @ PIRADS ≥ 3 AND AI at operrating point
    op_name = 'Optimized thd'
    threshold = operating_points['Optimized thd']
    if threshold is not None:
        plt.scatter((1-roc_data["Both_AI_and_Radiologist_Score"][(op_name, threshold)]["Specificity (TNR)"]),
                    roc_data["Both_AI_and_Radiologist_Score"][(
                        op_name, threshold)]["Sensitivity (TPR)"],
                    label=f'R@PIRADS≥3 AND AI@{threshold}', color='y', marker='+', s=100)

    # Plot AI OR Radiologist ROC and operating points
    plt.plot(fpr_or, tpr_or,
             label=f'Radiologist OR AI AUC = {roc_auc_or * 100:.2f}%', color='g', linestyle=':', linewidth=2)

    # OR Radiologist @ PIRADS ≥ 3 OR AI at operating point
    op_name = 'Optimized thd'
    threshold = operating_points['Optimized thd']
    if threshold is not None:
        plt.scatter((1-roc_data["Either_AI_or_Radiologist_Score"][(op_name, threshold)]["Specificity (TNR)"]),
                    roc_data["Either_AI_or_Radiologist_Score"][(
                        op_name, threshold)]["Sensitivity (TPR)"],
                    label=f'R@PIRADS≥3 OR AI@{threshold}', color='g', marker='x', s=100)

    # Finalize main plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(
        'ROC Curves for AI, Radiologist, and Combined Methods with Operating Points')
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(f'{output_dir}/roc_ai_radiologist_combined.png')
    plt.close()

    # Additional plot with only AI ROC and specific operating points
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_ai, tpr_ai,
             label=f'AI (continuous) AUC = {roc_auc_ai * 100:.2f}%', color='b', linewidth=2)
    for op_name, threshold in operating_points.items():
        if threshold is not None:
            plt.scatter((1-roc_data["AI"]["Operating Points"][(op_name, threshold)]["Specificity (TNR)"]),
                        roc_data["AI"]["Operating Points"][(
                            op_name, threshold)]["Sensitivity (TPR)"],
                        label=f'AI @ {threshold} - {op_name}', s=100)

    # Radiologist operating points
    plt.scatter(1 - metrics_radiologist_3["Specificity (TNR)"], metrics_radiologist_3["Sensitivity (TPR)"],
                label='Radiologist @ PIRADS ≥ 3', color='r', marker='^', s=100)
    plt.scatter(1 - metrics_radiologist_4["Specificity (TNR)"], metrics_radiologist_4["Sensitivity (TPR)"],
                label='Radiologist @ PIRADS ≥ 4', color='g', marker='^', s=100)

    # Finalize additional plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Participant-level ROC curve for software and radiologist')
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(f'{output_dir}/roc_ai_selected_operating_points.png')
    plt.close()

    # Calculate 95% CI for AI AUC using bootstrap
    auc_ci_ai = bootstrap_auc_ci(
        np.array(ground_truth), np.array(highest_ai_score))

    # Calculate 95% CI for Radiologist AUC using bootstrap
    auc_ci_radiologist = bootstrap_auc_ci(
        np.array(ground_truth), np.array(highest_pirads))

    # Log results in the log file
    with open(log_file, 'a') as log:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n\n<<< Patient-level Evaluation >>>\n")
        log.write(f"Timestamp: {current_time}\n")

        # Log overall AI AUC with CI
        log.write("\nSource: AI\n")
        log.write(f"  AUC: {roc_auc_ai:.4f}\n")
        log.write(
            f"  95% Confidence Interval for AUC: ({auc_ci_ai[0]:.4f}, {auc_ci_ai[1]:.4f})\n")

        # List of metrics to calculate their 95CI
        metric_name_list = ["Sensitivity (TPR)", "Specificity (TNR)", "Accuracy", "Precision", "F1 Score", "True Positives Percentage",
                            "False Positives Percentage", "True Negatives Percentage", "False Negatives Percentage"]
        # Log metrics for each AI operating point
        if "Operating Points" in roc_data["AI"]:
            log.write("\nAI Operating Points:\n")
            for op_name, metrics in roc_data["AI"]["Operating Points"].items():
                log.write(f"  Operating Point: {op_name}\n")
                for metric_name, metric_value in metrics.items():
                    log.write(f"    {metric_name}: {metric_value:.4f}\n")
                    # Add 95% CI for percentage metrics
                    if metric_name in metric_name_list:
                        total_cases = metrics["Positives Count"] + \
                            metrics["Negatives Count"]
                        ci = calculate_confidence_interval(
                            metric_value * 100, total_cases)
                        log.write(
                            f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")

        # Log Radiologist AUC with CI
        log.write("\nSource: Radiologist\n")
        log.write(f"  AUC: {roc_auc_radiologist:.4f}\n")
        log.write(
            f"  95% Confidence Interval for AUC: ({auc_ci_radiologist[0]:.4f}, {auc_ci_radiologist[1]:.4f})\n")

        # Log metrics for Radiologists at PIRADS thresholds
        log.write(f"\nSource: Radiologist @ PIRADS 3\n")
        for metric_name, metric_value in roc_data["Radiologist @ PIRADS ≥ 3"].items():
            log.write(f"  {metric_name}: {metric_value:.4f}\n")
            # Add 95% CI for percentage metrics
            if metric_name in metric_name_list:
                total_cases = metrics_radiologist_3["Positives Count"] + \
                    metrics_radiologist_3["Negatives Count"]
                ci = calculate_confidence_interval(
                    metric_value * 100, total_cases)
                log.write(
                    f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")

        log.write(f"\nSource: Radiologist @ PIRADS 4\n")
        for metric_name, metric_value in roc_data["Radiologist @ PIRADS ≥ 4"].items():
            log.write(f"  {metric_name}: {metric_value:.4f}\n")
            # Add 95% CI for percentage metrics
            if metric_name in metric_name_list:
                total_cases = metrics_radiologist_4["Positives Count"] + \
                    metrics_radiologist_4["Negatives Count"]
                ci = calculate_confidence_interval(
                    metric_value * 100, total_cases)
                log.write(
                    f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")

        # Log combined methods for "AI and Radiologist" and "AI or Radiologist"
        for method_label in ["Both_AI_and_Radiologist_Score", "Either_AI_or_Radiologist_Score"]:
            log.write(f"\nSource: {method_label}\n")
            for op_name, metrics in roc_data[method_label].items():
                log.write(f"  Operating Point: {op_name}\n")
                for metric_name, metric_value in metrics.items():
                    log.write(f"    {metric_name}: {metric_value:.4f}\n")
                    # Add 95% CI for percentage metrics
                    if metric_name in metric_name_list:
                        total_cases = metrics["Positives Count"] + \
                            metrics["Negatives Count"]
                        ci = calculate_confidence_interval(
                            metric_value * 100, total_cases)
                        log.write(
                            f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")

        # Log DeLong's test p-value
        log.write(
            f"\nDeLong's test p-value comparing AI and Radiologist: {p_value:.4f}\n")
        log.write("-------------------------------------------------\n")

    return roc_data, operating_points
