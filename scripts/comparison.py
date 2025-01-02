"""
Acquisition parameters code, part of the
Aalysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import numpy as np
from datetime import datetime
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests


def perform_wilcoxon_and_correct(data_pairs, fdr=0.05):
    """
    Perform Wilcoxon rank-sum tests on multiple pairs of data, collect p-values, and apply Benjamini-Hochberg correction.

    Parameters:
        data_pairs (list of tuples): A list of tuples, where each tuple contains two datasets to compare.
        fdr (float): False discovery rate for Benjamini-Hochberg correction (default is 0.05).

    Returns:
        dict: A dictionary containing:
              - 'raw_p_values': List of raw p-values from the Wilcoxon rank-sum tests.
              - 'corrected_p_values': List of p-values after Benjamini-Hochberg correction.
              - 'significant_tests': List of indices where tests are significant after correction.
    """
    raw_p_values = []

    # Perform Wilcoxon rank-sum test for each pair
    for idx, (data1, data2) in enumerate(data_pairs):
        stat, p_value = ranksums(data1, data2)
        raw_p_values.append(p_value)

    # Handle cases with only one p-value
    if len(raw_p_values) == 1:
        return {
            'raw_p_values': raw_p_values,
            'corrected_p_values': raw_p_values,
            'significant_tests': [0] if raw_p_values[0] < fdr else []
        }

    # Apply Benjamini-Hochberg correction
    _, corrected_p_values, _, _ = multipletests(
        raw_p_values, alpha=fdr, method='fdr_bh')

    return {
        'raw_p_values': raw_p_values,
        'corrected_p_values': corrected_p_values}


def run_comparisons(original_findings_statistics, adjusted_findings_statistics, log_file="comparisons_results.log", fdr=0.05):
    """
    Perform Wilcoxon rank-sum tests, log the results to a file, and apply Benjamini-Hochberg correction.

    Parameters:
        original_findings_statistics (dict): Statistics before adjustment.
        adjusted_findings_statistics (dict): Statistics after adjustment.
        log_file (str): Path to the log file to store results.
        fdr (float): False discovery rate for Benjamini-Hochberg correction (default is 0.05).

    Returns:
        results (dict): Results of the comparisons (p-values, p-values corrected, and significant or not).
    """
    # Extract datasets for comparisons
    # 1. Has Radiological findings [Original vs. Adjusted] (boolean)
    has_radiological_finding_original = np.array(
        original_findings_statistics["combined_stats"]["patients_with_findings_list"], dtype=int)
    has_radiological_finding_adjusted = np.array(
        adjusted_findings_statistics["combined_stats"]["patients_with_findings_list"], dtype=int)

    # 2. Highest AI probability scores [Original vs. Adjusted] (float)
    ai_highest_score_original = np.array(
        original_findings_statistics["ai_stats"]["highest_scores"])
    ai_highest_score_adjusted = np.array(
        adjusted_findings_statistics["ai_stats"]["highest_scores"])

    # 3. Number of lesions on bpMRI from AI [Original vs. Adjusted] (int)
    ai_findings_distribution_original = np.array(
        original_findings_statistics["ai_stats"]["findings_count"])
    ai_findings_distribution_adjusted = np.array(
        adjusted_findings_statistics["ai_stats"]["findings_count"])

    # 4. Number of lesions on bpMRI from combined [both Raiologist and AI] (float)
    combined_findings_distribution_original = np.array(
        original_findings_statistics["combined_stats"]["findings_count"], dtype=int)
    combined_findings_distribution_adjusted = np.array(
        adjusted_findings_statistics["combined_stats"]["findings_count"], dtype=int)

    # 5. Biopsy method [Original vs. Adjusted] (float)
    biopsy_method_distribution_original = np.array(
        original_findings_statistics["biopsy_stats"]["biopsy_method_list"], dtype=int)
    biopsy_method_distribution_adjusted = np.array(
        adjusted_findings_statistics["biopsy_stats"]["biopsy_method_list"], dtype=int)

    # 6. Highest ISUP GGG [Original vs. Adjusted] (float)
    ggg_higest_score_original = np.array(
        original_findings_statistics["ggg_stats"]["overall_highest_GGG_list"], dtype=int)
    ggg_higest_score_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["overall_highest_GGG_list"], dtype=int)

    # 7. Has combined (Human + AI Together) findings [Original vs. Adjusted] (boolean)
    has_combined_finding_original = np.array(
        original_findings_statistics["patient_findings_source_stats"][
            "patient_specific_lists"]["has_human_ai_together"], dtype=int)
    has_combined_finding_adjusted = np.array(
        adjusted_findings_statistics["patient_findings_source_stats"][
            "patient_specific_lists"]["has_human_ai_together"], dtype=int)

    # 8. Has only Human Alone findings [Original vs. Adjusted] (boolean)
    has_human_alone_finding_original = np.array(
        original_findings_statistics["patient_findings_source_stats"][
            "patient_specific_lists"]["has_human_alone"], dtype=int)
    has_human_alone_finding_adjusted = np.array(
        adjusted_findings_statistics["patient_findings_source_stats"][
            "patient_specific_lists"]["has_human_alone"], dtype=int)

    # 9. Has only AI Alone findings [Original vs. Adjusted] (boolean)
    has_ai_alone_finding_original = np.array(
        original_findings_statistics["patient_findings_source_stats"][
            "patient_specific_lists"]["has_ai_alone"], dtype=int)
    has_ai_alone_finding_adjusted = np.array(
        adjusted_findings_statistics["patient_findings_source_stats"][
            "patient_specific_lists"]["has_ai_alone"], dtype=int)

    # 10. Patient has overall significant cancer or not [Original vs. Adjusted] (int)
    patient_has_overall_cspca_original = np.array(
        original_findings_statistics["ggg_stats"]["patient_has_overall_cspca"])
    patient_has_overall_cspca_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["patient_has_overall_cspca"])

    # 11. Patient don't have overall significant cancer or not [Original vs. Adjusted] (int)
    patient_has_overall_non_cspca_original = np.array(
        original_findings_statistics["ggg_stats"]["patient_has_overall_non_cspca"])
    patient_has_overall_non_cspca_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["patient_has_overall_non_cspca"])

    # 12. Patient has AI significant cancer or not [Original vs. Adjusted] (int)
    patient_has_ai_cspca_original = np.array(
        original_findings_statistics["ggg_stats"]["significant_ai_patients"])
    patient_has_ai_cspca_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["significant_ai_patients"])

    # 13. Patient has AI non-significant cancer or not [Original vs. Adjusted] (int)
    patient_has_ai_non_cspca_original = np.array(
        original_findings_statistics["ggg_stats"]["non_significant_ai_patients"])
    patient_has_ai_non_cspca_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["non_significant_ai_patients"])

    # 14. Original Patient has significant cancer or not [from Human vs from AI] (int)
    patient_has_cspca_from_human_original = np.array(
        original_findings_statistics["ggg_stats"]["significant_human_patients"])
    patient_has_cspca_from_ai_original = np.array(
        original_findings_statistics["ggg_stats"]["significant_ai_patients"])

    # 15. Original Patient has non-significant cancer or not [from Human vs from AI] (int)
    patient_has_non_cspca_from_human_original = np.array(
        original_findings_statistics["ggg_stats"]["non_significant_human_patients"])
    patient_has_non_cspca_from_ai_original = np.array(
        original_findings_statistics["ggg_stats"]["non_significant_ai_patients"])

    # 16. Adjusted Patient has significant cancer or not [from Human vs from AI] (int)
    patient_has_cspca_from_human_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["significant_human_patients"])
    patient_has_cspca_from_ai_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["significant_ai_patients"])

    # 17. Adjusted Patient has non-significant cancer or not [from Human vs from AI] (int)
    patient_has_non_cspca_from_human_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["non_significant_human_patients"])
    patient_has_non_cspca_from_ai_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["non_significant_ai_patients"])

    # 18. Lesion has AI significant cancer or not [Original vs. Adjusted] (int)
    lesion_has_ai_cspca_original = np.array(
        original_findings_statistics["ggg_stats"]["ai_lesion_has_cspca"])
    lesion_has_ai_cspca_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["ai_lesion_has_cspca"])

    # 19. Lesion has AI non-significant cancer or not [Original vs. Adjusted] (int)
    lesion_has_ai_non_cspca_original = np.array(
        original_findings_statistics["ggg_stats"]["ai_lesion_has_non_cspca"])
    lesion_has_ai_non_cspca_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["ai_lesion_has_non_cspca"])

    # 20. Original Lesion has significant cancer or not [from Human vs from AI] (int)
    lesion_has_cspca_from_human_original = np.array(
        original_findings_statistics["ggg_stats"]["significant_human_lesions"])
    lesion_has_cspca_from_ai_original = np.array(
        original_findings_statistics["ggg_stats"]["significant_ai_lesions"])

    # 21. Original Lesion has non-significant cancer or not [from Human vs from AI] (int)
    lesion_has_non_cspca_from_human_original = np.array(
        original_findings_statistics["ggg_stats"]["non_significant_human_lesions"])
    lesion_has_non_cspca_from_ai_original = np.array(
        original_findings_statistics["ggg_stats"]["non_significant_ai_lesions"])

    # 22. Adjusted Lesion has significant cancer or not [from Human vs from AI] (int)
    lesion_has_cspca_from_human_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["significant_human_lesions"])
    lesion_has_cspca_from_ai_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["significant_ai_lesions"])

    # 23. Adjusted Lesion has non-significant cancer or not [from Human vs from AI] (int)
    lesion_has_non_cspca_from_human_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["non_significant_human_lesions"])
    lesion_has_non_cspca_from_ai_adjusted = np.array(
        adjusted_findings_statistics["ggg_stats"]["non_significant_ai_lesions"])

    # Construct data pairs list
    data_pairs = [
        (has_radiological_finding_original, has_radiological_finding_adjusted),
        (ai_highest_score_original, ai_highest_score_adjusted),
        (ai_findings_distribution_original, ai_findings_distribution_adjusted),
        (combined_findings_distribution_original,
         combined_findings_distribution_adjusted),
        (biopsy_method_distribution_original, biopsy_method_distribution_adjusted),
        (ggg_higest_score_original, ggg_higest_score_adjusted),
        (has_combined_finding_original, has_combined_finding_adjusted),
        (has_human_alone_finding_original,
         has_human_alone_finding_adjusted),
        (has_ai_alone_finding_original, has_ai_alone_finding_adjusted),
        (patient_has_overall_cspca_original, patient_has_overall_cspca_adjusted),
        (patient_has_overall_non_cspca_original,
         patient_has_overall_non_cspca_adjusted),
        (patient_has_ai_cspca_original, patient_has_ai_cspca_adjusted),
        (patient_has_ai_non_cspca_original, patient_has_ai_non_cspca_adjusted),
        (patient_has_cspca_from_human_original,
         patient_has_cspca_from_ai_original),
        (patient_has_non_cspca_from_human_original,
         patient_has_non_cspca_from_ai_original),
        (patient_has_cspca_from_human_adjusted,
         patient_has_cspca_from_ai_adjusted),
        (patient_has_non_cspca_from_human_adjusted,
         patient_has_non_cspca_from_ai_adjusted),
        (lesion_has_ai_cspca_original, lesion_has_ai_cspca_adjusted),
        (lesion_has_ai_non_cspca_original, lesion_has_ai_non_cspca_adjusted),
        (lesion_has_cspca_from_human_original, lesion_has_cspca_from_ai_original),
        (lesion_has_non_cspca_from_human_original,
         lesion_has_non_cspca_from_ai_original),
        (lesion_has_cspca_from_human_adjusted, lesion_has_cspca_from_ai_adjusted),
        (lesion_has_non_cspca_from_human_adjusted,
         lesion_has_non_cspca_from_ai_adjusted)
    ]

    # Run tests and BH correction
    results = perform_wilcoxon_and_correct(data_pairs, fdr)

    # Log results
    with open(log_file, "a") as log:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n<<< Comparisions Results >>>\n")
        log.write(f"\n<<< Wilcoxon Rank-Sum Test Results >>>\n")
        log.write(f"Timestamp: {current_time}\n")
        log.write("--------------------------------\n")

        # Add specific context to each test
        test_descriptions = [
            "1. Has Radiological findings: Before vs After adjustment",
            "2. AI highest scores: Before vs After adjustment",
            "3. Number of lesions distribution from AI: Before vs After adjustment",
            "4. Number of lesions distribution from combined: Before vs After adjustment",
            "5. Biopsy method distribution: Before vs After adjustment",
            "6. GGG highest scores: Before vs After adjustment",
            "7. Has Combined findings: Before vs After adjustment",
            "8. Has Human alone findings: Before vs After adjustment",
            "9. Has AI alone findings: Before vs After adjustment",
            "10. Patient has overall csPCa: Before vs After adjustment",
            "11. Patient has vcerall non-csPCa: Before vs After adjustment",
            "12. Patient has AI csPCa: Before vs After adjustment",
            "13. Patient has AI non-csPCa: Before vs After adjustment",
            "14. Patient has Human csPC from Human vs from AI: Original",
            "15. Patient has Human non-csPC from Human vs from AI: Original",
            "16. Patient has Human csPC from Human vs from AI: Adjusted",
            "17. Patient has Human non-csPC from Human vs from AI: Adjusted",
            "18. Lesion has AI csPCa: Before vs After adjustment",
            "19. Lesion has AI non-csPCa: Before vs After adjustment",
            "20. Lesion has csPCa from Human vs from AI: Original",
            "21. Lesion has non-csPCa from Human vs from AI: Original",
            "22. Lesion has csPCa from Human vs from AI: Adjusted",
            "23. Lesion has non-csPCa from Human vs from AI: Adjusted"
        ]

        for idx, (description, raw_p, corrected_p) in enumerate(zip(test_descriptions, results['raw_p_values'], results['corrected_p_values'])):
            log.write(
                f"Test {idx + 1}: {description}\n"
                f"    Raw p-value = {raw_p:.4e}\n"
                f"    Corrected p-value = {corrected_p:.4e}\n"
            )

    return results
