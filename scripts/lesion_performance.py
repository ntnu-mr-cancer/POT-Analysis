"""
Lesion-level performance analysis code, part of the
Analysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import calculate_confidence_interval, bootstrap_ap_ci
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, f1_score,
    precision_recall_curve, average_precision_score, recall_score
)


def evaluate_lesion_level_performance(prepared_data, operating_points, log_file="lesion_level_performance.log", output_dir='plots'):
    """
    Evaluates lesion-level performance of AI and Radiologists in predicting significant prostate cancer.
    Includes combined approaches ("AI or Radiologist", "AI and Radiologist") and generates FROC and AP plots.

    Parameters:
        prepared_data (pd.DataFrame): Prepared dataset to analyze.
        operating_points (dict): AI thresholds to evaluate, with 'Optimized thd' as main cut-off.
        log_file (str): Path to the log file for results.
        output_dir (str): Directory to save plots.

    Returns:
        lesions_df (pd.DataFrame): Data frame with lesions details from lesion analysis.
        roc_data (dict): Performance metrics and lesion-level data.
    """
    # Identify thrshoulds
    # Radiologist thresholds
    RADIOLOGIST_THRESHOLD_3 = 3
    RADIOLOGIST_THRESHOLD_4 = 4

    # Main cut-off is `Optimized thd` from operating_points
    MAIN_AI_THRESHOLD = operating_points['Optimized thd']

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to convert GGG values to numeric, handling "0 (<1)" cases
    def convert_to_numeric(value):
        if value is None or value == "0 (<1)":
            return 0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    # Extract lesions with associated AI and Radiologist scores from prepared data
    def extract_lesions_and_scores(prepared_data):
        all_lesions = []

        for index, row in prepared_data.iterrows():
            num_targeted = int(row['NumberofTargetedBiopsies'])
            num_systematic = int(row['NumberofSystematicBiopsies'])

            # Process targeted biopsies
            for i in range(1, num_targeted + 1):
                lesion_col = f'LesionOverallGGGTargeted{i}'
                ai_col = f'AIScore{i}'
                pirads_col = 'PIRADS' if i == 1 else f'PIRADS{i}'

                # Convert values, default to 0 if missing or NaN
                lesion = convert_to_numeric(row.get(lesion_col, 0))
                ai_score = pd.to_numeric(
                    row.get(ai_col, 0), errors='coerce') or 0
                radiologist_score = pd.to_numeric(
                    row.get(pirads_col, 0), errors='coerce') or 0
                true_label = 1 if lesion >= 2 else 0

                all_lesions.append({
                    'Patient_ID': row['PatientID'],
                    'LesionType': 'Targeted',
                    'True_Label': true_label,
                    'AI_Score': ai_score if not pd.isna(ai_score) else 0,
                    'Radiologist_Score': radiologist_score if not pd.isna(radiologist_score) else 0,
                    'Both_AI_and_Radiologist_Score': ((ai_score >= MAIN_AI_THRESHOLD) & (radiologist_score > RADIOLOGIST_THRESHOLD_3)).astype(int) if not pd.isna(ai_score) else 0,
                    'Either_AI_or_Radiologist_Score': ((ai_score >= MAIN_AI_THRESHOLD) | (radiologist_score > RADIOLOGIST_THRESHOLD_3)).astype(int) if not pd.isna(ai_score) else 0
                })

            # Process systematic biopsies (no AI or Radiologist scores)
            for i in range(1, num_systematic + 1):
                lesion_col = f'BiopsyOverallGGGSystematic{i}'
                lesion = convert_to_numeric(row.get(lesion_col, 0))
                true_label = 1 if lesion >= 2 else 0

                # only if positive record it
                if true_label:
                    all_lesions.append({
                        'Patient_ID': row['PatientID'],
                        'LesionType': 'Systematic',
                        'True_Label': true_label,
                        'AI_Score': 0,
                        'Radiologist_Score': 0,
                        'Both_AI_and_Radiologist_Score': 0,
                        'Either_AI_or_Radiologist_Score': 0
                    })

        return pd.DataFrame(all_lesions)

    # Extract lesions and fill missing values with 0
    lesions_df = extract_lesions_and_scores(prepared_data)

    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'Sensitivity (TPR)': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity (TNR)': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
            'FPPI': fp / len(np.unique(prepared_data['PatientID'])),
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

    # Initialize dictionary for storing metrics data
    roc_data = {'AI': {}, 'Radiologist': {},
                'Either_AI_or_Radiologist_Score': {}, 'Both_AI_and_Radiologist_Score': {}}

    # Calculate and log metrics for AI at each operating point
    for op_name, threshold in operating_points.items():
        ai_predictions = (lesions_df['AI_Score'] >= threshold).astype(int)
        metrics = calculate_metrics(lesions_df['True_Label'], ai_predictions)
        metrics['Threshold'] = threshold  # Ensure 'Threshold' is added
        roc_data['AI'][op_name] = metrics

    # Calculate metrics for Radiologist at PIRADS thresholds (3 and 4)
    for threshold, label in [(RADIOLOGIST_THRESHOLD_3, 'PIRADS 3'), (RADIOLOGIST_THRESHOLD_4, 'PIRADS 4')]:
        radiologist_predictions = (
            lesions_df['Radiologist_Score'] >= threshold).astype(int)
        metrics = calculate_metrics(
            lesions_df['True_Label'], radiologist_predictions)
        metrics['Threshold'] = threshold  # Ensure 'Threshold' is added
        roc_data['Radiologist'][label] = metrics

    # Calculate metrics for "AI or Radiologist" and "AI and Radiologist" at each operating point
    op_name = 'Optimized thd'
    threshold = operating_points['Optimized thd']
    if threshold is not None:
        or_predictions = (
            lesions_df['Either_AI_or_Radiologist_Score'] >= threshold).astype(int)
        and_predictions = (
            lesions_df['Both_AI_and_Radiologist_Score'] >= threshold).astype(int)

        metrics_or = calculate_metrics(
            lesions_df['True_Label'], or_predictions)
        metrics_or['Threshold'] = threshold  # Ensure 'Threshold' is added
        roc_data['Either_AI_or_Radiologist_Score'][op_name] = metrics_or

        metrics_and = calculate_metrics(
            lesions_df['True_Label'], and_predictions)
        metrics_and['Threshold'] = threshold  # Ensure 'Threshold' is added
        roc_data['Both_AI_and_Radiologist_Score'][op_name] = metrics_and

    # Helper function to compute FROC curve for combined methods
    def compute_combined_froc(scores, patient_ids, true_labels, threshold):
        sensitivities, avg_false_positives = [], []
        thresholds = np.sort(np.unique(scores[scores > 0]))

        for thresh in thresholds:
            predictions = (scores >= thresh).astype(int)
            tp = np.sum(predictions & true_labels)
            fp = np.sum(predictions & ~true_labels)
            fn = np.sum(~predictions & true_labels)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            avg_fp_per_patient = fp / \
                len(np.unique(prepared_data['PatientID']))

            sensitivities.append(sensitivity)
            avg_false_positives.append(avg_fp_per_patient)

        return avg_false_positives, sensitivities

    # FROC Plot - AI, AI and Radiologist, and AI or Radiologist with Operating Points
    avg_fp_ai, sensitivities_ai = compute_combined_froc(
        lesions_df['AI_Score'], lesions_df['Patient_ID'], lesions_df['True_Label'], MAIN_AI_THRESHOLD
    )

    plt.figure(figsize=(10, 8))
    plt.plot(avg_fp_ai, sensitivities_ai, linestyle='-',
             color='b', label='AI (continuous)', linewidth=2)

    # Mark AI operating points
    for op_name, threshold in operating_points.items():
        ai_predictions = (lesions_df['AI_Score'] >= threshold).astype(int)
        avg_fp_op = calculate_metrics(
            lesions_df['True_Label'], ai_predictions)['FPPI']
        sensitivity_op = calculate_metrics(lesions_df['True_Label'], ai_predictions)[
            'Sensitivity (TPR)']
        plt.scatter(avg_fp_op, sensitivity_op, s=100,
                    label=f'AI @ {threshold} - {op_name}', marker='o')

    # Mark Radiologist operating points
    threshold = RADIOLOGIST_THRESHOLD_3
    label = 'PIRADS≥3'
    radiologist_predictions = (
        lesions_df['Radiologist_Score'] >= threshold).astype(int)
    avg_fp_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['FPPI']
    sensitivity_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['Sensitivity (TPR)']
    plt.scatter(avg_fp_op, sensitivity_op, s=100,
                label=f'Radiologist @ {label}', marker='^', color='r')

    threshold = RADIOLOGIST_THRESHOLD_4
    label = 'PIRADS≥4'
    radiologist_predictions = (
        lesions_df['Radiologist_Score'] >= threshold).astype(int)
    avg_fp_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['FPPI']
    sensitivity_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['Sensitivity (TPR)']
    plt.scatter(avg_fp_op, sensitivity_op, s=100,
                label=f'Radiologist @ {label}', marker='^', color='g')

    # Mark Radiologist AND AI operating point
    op_name = 'Optimized thd'
    threshold = operating_points['Optimized thd']
    if threshold is not None:
        avg_fp_op_and = roc_data["Both_AI_and_Radiologist_Score"][op_name]["FPPI"]
        sensitivity_op_and = roc_data["Both_AI_and_Radiologist_Score"][
            op_name]["Sensitivity (TPR)"]
        plt.scatter(avg_fp_op_and, sensitivity_op_and, s=100,
                    label=f'R@PIRADS≥3 AND AI@{threshold}', marker='+', color='y')

        avg_fp_op_or = roc_data["Either_AI_or_Radiologist_Score"][op_name]["FPPI"]
        sensitivity_op_or = roc_data["Either_AI_or_Radiologist_Score"][
            op_name]["Sensitivity (TPR)"]
        plt.scatter(avg_fp_op_or, sensitivity_op_or, s=100,
                    label=f'R@PIRADS≥3 OR AI@{threshold}', marker='x', color='g')

    plt.xlabel('Average Number of False Positives per Patient (FPPI)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('FROC Curve - AI, AI and Radiologist, AI or Radiologist')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(
        f'{output_dir}/FROC_Curve_All_Methods_with_Operating_Points.png')
    plt.show()

    # Simple figure
    plt.figure(figsize=(10, 8))
    plt.plot(avg_fp_ai, sensitivities_ai, linestyle='-',
             color='b', label='AI (continuous)', linewidth=2)

    # Mark AI operating points
    for op_name, threshold in operating_points.items():
        ai_predictions = (lesions_df['AI_Score'] >= threshold).astype(int)
        avg_fp_op = calculate_metrics(
            lesions_df['True_Label'], ai_predictions)['FPPI']
        sensitivity_op = calculate_metrics(lesions_df['True_Label'], ai_predictions)[
            'Sensitivity (TPR)']
        plt.scatter(avg_fp_op, sensitivity_op, s=100,
                    label=f'AI @ {threshold} - {op_name}', marker='o')

    # Mark Radiologist operating points
    threshold = RADIOLOGIST_THRESHOLD_3
    label = 'PIRADS≥3'
    radiologist_predictions = (
        lesions_df['Radiologist_Score'] >= threshold).astype(int)
    avg_fp_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['FPPI']
    sensitivity_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['Sensitivity (TPR)']
    plt.scatter(avg_fp_op, sensitivity_op, s=100,
                label=f'Radiologist @ {label}', marker='^', color='r')

    threshold = RADIOLOGIST_THRESHOLD_4
    label = 'PIRADS≥4'
    radiologist_predictions = (
        lesions_df['Radiologist_Score'] >= threshold).astype(int)
    avg_fp_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['FPPI']
    sensitivity_op = calculate_metrics(
        lesions_df['True_Label'], radiologist_predictions)['Sensitivity (TPR)']
    plt.scatter(avg_fp_op, sensitivity_op, s=100,
                label=f'Radiologist @ {label}', marker='^', color='g')

    plt.xlabel('Average Number of False Positives per Patient (FPPI)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Lesion-level FROC Curve for software and radiologist')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(
        f'{output_dir}/FROC_Curve_ai_radiologist_with_Operating_Points.png')
    plt.show()

    # AP Plot - AI, Radiologist, AI or Radiologist, AI and Radiologist
    def plot_precision_recall_all(lesions_df, output_dir):
        # Calculate precision-recall for each method
        precision_ai, recall_ai, _ = precision_recall_curve(
            lesions_df['True_Label'], lesions_df['AI_Score'])
        ap_score_ai = average_precision_score(
            lesions_df['True_Label'], lesions_df['AI_Score'])

        precision_ai_or, recall_ai_or, _ = precision_recall_curve(
            lesions_df['True_Label'], lesions_df['Either_AI_or_Radiologist_Score']
        )
        ap_score_ai_or = average_precision_score(
            lesions_df['True_Label'], lesions_df['Either_AI_or_Radiologist_Score'])

        precision_ai_and, recall_ai_and, _ = precision_recall_curve(
            lesions_df['True_Label'], lesions_df['Both_AI_and_Radiologist_Score']
        )
        ap_score_ai_and = average_precision_score(
            lesions_df['True_Label'], lesions_df['Both_AI_and_Radiologist_Score'])

        precision_radiologist, recall_radiologist, _ = precision_recall_curve(
            lesions_df['True_Label'], lesions_df['Radiologist_Score']
        )
        ap_score_radiologist = average_precision_score(
            lesions_df['True_Label'], lesions_df['Radiologist_Score'])

        # Plotting AP Curve with all methods
        plt.figure(figsize=(10, 8))
        plt.step(recall_ai, precision_ai, where='post', linestyle='-',
                 color='b', linewidth=2, label=f'AI AP = {ap_score_ai * 100:.2f}%')
        plt.step(recall_ai_or, precision_ai_or, where='post', linestyle=':', color='g',
                 linewidth=2, label=f'Either AI or Radiologisst AP = {ap_score_ai_or * 100:.2f}%')
        plt.step(recall_ai_and, precision_ai_and, where='post', linestyle='-.', color='y',
                 linewidth=2, label=f'Both AI and Radiologist AP = {ap_score_ai_and * 100:.2f}%')
        plt.step(recall_radiologist, precision_radiologist, where='post', linestyle='--',
                 color='r', linewidth=2, label=f'Radiologist AP = {ap_score_radiologist * 100:.2f}%')

        # Mark AI operating points on the AP plot
        for op_name, threshold in operating_points.items():
            ai_predictions = (lesions_df['AI_Score'] >= threshold)
            recall_op = recall_score(lesions_df['True_Label'], ai_predictions)
            precision_op = precision_score(
                lesions_df['True_Label'], ai_predictions, zero_division=0)
            plt.scatter(recall_op, precision_op, s=100,
                        label=f'AI @ {threshold} - {op_name}', marker='o')

        # Mark Radiologist operating points
        threshold = RADIOLOGIST_THRESHOLD_3
        label = 'PIRADS≥3'
        radiologist_predictions = (
            lesions_df['Radiologist_Score'] >= threshold)
        recall_op = recall_score(
            lesions_df['True_Label'], radiologist_predictions)
        precision_op = precision_score(
            lesions_df['True_Label'], radiologist_predictions, zero_division=0)
        plt.scatter(recall_op, precision_op, s=100,
                    label=f'Radiologist @ {label}', marker='^', color='r')

        threshold = RADIOLOGIST_THRESHOLD_4
        label = 'PIRADS≥4'
        radiologist_predictions = (
            lesions_df['Radiologist_Score'] >= threshold)
        recall_op = recall_score(
            lesions_df['True_Label'], radiologist_predictions)
        precision_op = precision_score(
            lesions_df['True_Label'], radiologist_predictions, zero_division=0)
        plt.scatter(recall_op, precision_op, s=100,
                    label=f'Radiologist @ {label}', marker='^', color='g')

        # Mark Radiologist AND AI operating point
        op_name = 'Optimized thd'
        threshold = operating_points['Optimized thd']
        if threshold is not None:
            and_predictions = (
                lesions_df['Both_AI_and_Radiologist_Score'] >= threshold)
            recall_op_and = recall_score(
                lesions_df['True_Label'], and_predictions)
            precision_op_and = precision_score(
                lesions_df['True_Label'], and_predictions, zero_division=0)
            plt.scatter(recall_op_and, precision_op_and, s=100,
                        label=f'R@PIRADS≥3 AND AI@{threshold}', marker='+', color='y')

            or_predictions = (
                lesions_df['Either_AI_or_Radiologist_Score'] >= threshold)
            recall_op_or = recall_score(
                lesions_df['True_Label'], or_predictions)
            precision_op_or = precision_score(
                lesions_df['True_Label'], or_predictions, zero_division=0)
            plt.scatter(recall_op_or, precision_op_or, s=100,
                        label=f'R@PIRADS≥3 OR AI@{threshold}', marker='x', color='g')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - All Methods')
        plt.legend(loc='upper right', prop={'size': 10}, frameon=True)
        plt.savefig(f'{output_dir}/AP_Curve_All_Methods.png')
        plt.show()

        return ap_score_ai, ap_score_ai_or, ap_score_ai_and, ap_score_radiologist

     # AP Plot - AI, Radiologist, AI or Radiologist, AI and Radiologist
    def plot_precision_recall_ai_radiologist(lesions_df, output_dir):
        # Calculate precision-recall for each method
        precision_ai, recall_ai, _ = precision_recall_curve(
            lesions_df['True_Label'], lesions_df['AI_Score'])
        ap_score_ai = average_precision_score(
            lesions_df['True_Label'], lesions_df['AI_Score'])

        precision_radiologist, recall_radiologist, _ = precision_recall_curve(
            lesions_df['True_Label'], lesions_df['Radiologist_Score']
        )
        ap_score_radiologist = average_precision_score(
            lesions_df['True_Label'], lesions_df['Radiologist_Score'])

        # Plotting AP Curve with all methods
        plt.figure(figsize=(10, 8))
        plt.step(recall_ai, precision_ai, where='post', linestyle='-',
                 color='b', linewidth=2, label=f'AI AP = {ap_score_ai * 100:.2f}%')
        plt.step(recall_radiologist, precision_radiologist, where='post', linestyle='--',
                 color='r', linewidth=2, label=f'Radiologist AP = {ap_score_radiologist * 100:.2f}%')

        # Mark AI operating points on the AP plot
        for op_name, threshold in operating_points.items():
            ai_predictions = (lesions_df['AI_Score'] >= threshold)
            recall_op = recall_score(lesions_df['True_Label'], ai_predictions)
            precision_op = precision_score(
                lesions_df['True_Label'], ai_predictions, zero_division=0)
            plt.scatter(recall_op, precision_op, s=100,
                        label=f'AI @ {threshold} - {op_name}', marker='o')

        # Mark Radiologist operating points
        threshold = RADIOLOGIST_THRESHOLD_3
        label = 'PIRADS≥3'
        radiologist_predictions = (
            lesions_df['Radiologist_Score'] >= threshold)
        recall_op = recall_score(
            lesions_df['True_Label'], radiologist_predictions)
        precision_op = precision_score(
            lesions_df['True_Label'], radiologist_predictions, zero_division=0)
        plt.scatter(recall_op, precision_op, s=100,
                    label=f'Radiologist @ {label}', marker='^', color='r')

        threshold = RADIOLOGIST_THRESHOLD_4
        label = 'PIRADS≥4'
        radiologist_predictions = (
            lesions_df['Radiologist_Score'] >= threshold)
        recall_op = recall_score(
            lesions_df['True_Label'], radiologist_predictions)
        precision_op = precision_score(
            lesions_df['True_Label'], radiologist_predictions, zero_division=0)
        plt.scatter(recall_op, precision_op, s=100,
                    label=f'Radiologist @ {label}', marker='^', color='g')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(
            'Lesion-level Precision-Recall Curve for software and radiologist')
        plt.legend(loc='upper right', frameon=True)
        plt.savefig(f'{output_dir}/AP_Curve_ai_radiologist.png')
        plt.show()

    # Call the AP plot function
    ap_score_ai, ap_score_ai_or, ap_score_ai_and, ap_score_radiologist = plot_precision_recall_all(
        lesions_df, output_dir)
    plot_precision_recall_ai_radiologist(lesions_df, output_dir)

    # Calculate 95% CIs for AP using bootstrapping
    ap_ci_ai = bootstrap_ap_ci(
        np.array(lesions_df['True_Label']), np.array(lesions_df['AI_Score']))
    ap_ci_radiologist = bootstrap_ap_ci(
        np.array(lesions_df['True_Label']), np.array(lesions_df['Radiologist_Score']))
    ap_ci_ai_or = bootstrap_ap_ci(np.array(lesions_df['True_Label']), np.array(
        lesions_df['Either_AI_or_Radiologist_Score']))
    ap_ci_ai_and = bootstrap_ap_ci(np.array(lesions_df['True_Label']), np.array(
        lesions_df['Both_AI_and_Radiologist_Score']))

    # Final logging of all metrics with a check for the 'Threshold' key
    with open(log_file, 'a') as log:
        log.write("\n\n<<< Lesion-level Evaluation >>>\n")

        # Log AI metrics at each operating point
        log.write("\nAI Operating Points Metrics:\n")
        log.write(f"AP = {ap_score_ai:.4f}\n")
        log.write(
            f"95% Confidence Interval for AP: ({ap_ci_ai[0]:.4f}, {ap_ci_ai[1]:.4f})\n")

        # List of metrics to caclulate 95% CI for
        metric_name_list = ["Sensitivity (TPR)", "Specificity (TNR)", "Accuracy", "Precision", "F1 Score", "True Positives Percentage",
                            "False Positives Percentage", "True Negatives Percentage", "False Negatives Percentage"]

        for op_name, metrics in roc_data['AI'].items():
            threshold = metrics.get('Threshold', 'N/A')
            log.write(
                f"  Operating Point: {op_name} (Threshold: {threshold})\n")
            for metric_name, metric_value in metrics.items():
                if metric_name != 'Threshold':
                    log.write(f"    {metric_name}: {metric_value:.4f}\n")
                    # Add 95% CI for percentage metrics
                    if metric_name in metric_name_list:
                        total_cases = metrics["Positives Count"] + \
                            metrics["Negatives Count"]
                        ci = calculate_confidence_interval(
                            metric_value * 100, total_cases)
                        log.write(
                            f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")
                    if metric_name == "FPPI":
                        patient_count = len(
                            np.unique(lesions_df['Patient_ID']))
                        ci_fppi = calculate_confidence_interval(
                            metric_value * 100, patient_count)
                        log.write(
                            f"    95% Confidence Interval for FPPI: ({ci_fppi[0]:.2f}%, {ci_fppi[1]:.2f}%)\n")

        # Log Radiologist metrics at PIRADS 3 and PIRADS 4
        log.write("\nRadiologist Metrics:\n")
        log.write(f"AP = {ap_score_radiologist:.4f}\n")
        log.write(
            f"95% Confidence Interval for AP: ({ap_ci_radiologist[0]:.4f}, {ap_ci_radiologist[1]:.4f})\n")
        for label, metrics in roc_data['Radiologist'].items():
            threshold = metrics.get('Threshold', 'N/A')
            log.write(f"  Threshold: {label} (Threshold: {threshold})\n")
            for metric_name, metric_value in metrics.items():
                if metric_name != 'Threshold':
                    log.write(f"    {metric_name}: {metric_value:.4f}\n")
                    if metric_name in metric_name_list:
                        total_cases = metrics["Positives Count"] + \
                            metrics["Negatives Count"]
                        ci = calculate_confidence_interval(
                            metric_value * 100, total_cases)
                        log.write(
                            f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")
                    if metric_name == "FPPI":
                        patient_count = len(
                            np.unique(lesions_df['Patient_ID']))
                        ci_fppi = calculate_confidence_interval(
                            metric_value * 100, patient_count)
                        log.write(
                            f"    95% Confidence Interval for FPPI: ({ci_fppi[0]:.2f}%, {ci_fppi[1]:.2f}%)\n")

        # Log "Either AI or Radiologist" metrics at each operating point
        log.write("\n'Either AI or Radiologist' Operating Points Metrics:\n")
        log.write(f"AP = {ap_score_ai_or:.4f}\n")
        log.write(
            f"95% Confidence Interval for AP: ({ap_ci_ai_or[0]:.4f}, {ap_ci_ai_or[1]:.4f})\n")
        for op_name, metrics in roc_data['Either_AI_or_Radiologist_Score'].items():
            threshold = metrics.get('Threshold', 'N/A')
            log.write(
                f"  Operating Point: {op_name} (Threshold: {threshold})\n")
            for metric_name, metric_value in metrics.items():
                if metric_name != 'Threshold':
                    log.write(f"    {metric_name}: {metric_value:.4f}\n")
                    if metric_name in metric_name_list:
                        total_cases = metrics["Positives Count"] + \
                            metrics["Negatives Count"]
                        ci = calculate_confidence_interval(
                            metric_value * 100, total_cases)
                        log.write(
                            f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")
                    if metric_name == "FPPI":
                        patient_count = len(
                            np.unique(lesions_df['Patient_ID']))
                        ci_fppi = calculate_confidence_interval(
                            metric_value * 100, patient_count)
                        log.write(
                            f"    95% Confidence Interval for FPPI: ({ci_fppi[0]:.2f}%, {ci_fppi[1]:.2f}%)\n")

        # Log "Both AI and Radiologist" metrics at each operating point
        log.write("\n'Both AI and Radiologist' Operating Points Metrics:\n")
        log.write(f"AP = {ap_score_ai_and:.4f}\n")
        log.write(
            f"95% Confidence Interval for AP: ({ap_ci_ai_and[0]:.4f}, {ap_ci_ai_and[1]:.4f})\n")
        for op_name, metrics in roc_data['Both_AI_and_Radiologist_Score'].items():
            threshold = metrics.get('Threshold', 'N/A')
            log.write(
                f"  Operating Point: {op_name} (Threshold: {threshold})\n")
            for metric_name, metric_value in metrics.items():
                if metric_name != 'Threshold':
                    log.write(f"    {metric_name}: {metric_value:.4f}\n")
                    if metric_name in metric_name_list:
                        total_cases = metrics["Positives Count"] + \
                            metrics["Negatives Count"]
                        ci = calculate_confidence_interval(
                            metric_value * 100, total_cases)
                        log.write(
                            f"    95% Confidence Interval: ({ci[0]:.2f}%, {ci[1]:.2f}%)\n")
                    if metric_name == "FPPI":
                        patient_count = len(
                            np.unique(lesions_df['Patient_ID']))
                        ci_fppi = calculate_confidence_interval(
                            metric_value * 100, patient_count)
                        log.write(
                            f"    95% Confidence Interval for FPPI: ({ci_fppi[0]:.2f}%, {ci_fppi[1]:.2f}%)\n")

    return lesions_df, roc_data
