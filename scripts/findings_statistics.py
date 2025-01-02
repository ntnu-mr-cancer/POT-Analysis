"""
Findings statistics code, part of the
Analysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import numpy as np
import pandas as pd
from scripts.utils import calculate_confidence_interval
from datetime import datetime


def log_to_file(log_file, section_title, content):
    """
    Helper function to append logs to the log file.

    Parameters:
        log_file (str): Path to the log file.
        section_title (str): Title of the section to log.
        content (str): Content to append under the section.
    """
    with open(log_file, "a") as log:
        log.write(f"\n\n<<< {section_title} >>>\n")
        log.write(content + "\n")
        log.write("-" * 100 + "\n")


def analyze_combined_findings(prepared_data):
    """
    Analyze combined radiological and AI findings to determine patient-level statistics.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with findings data.

    Returns:
        stats (dict): Summary statistics for combined findings, including a table for findings number distribution.
    """
    # Convert columns to numeric, handling errors
    prepared_data["FindingsNumber"] = pd.to_numeric(
        prepared_data["FindingsNumber"], errors="coerce")
    prepared_data["NumberFindingsRejected"] = pd.to_numeric(
        prepared_data["NumberFindingsRejected"], errors="coerce")

    # Identify patients with findings
    combined_findings = prepared_data["FindingsNumber"] > 0

    # Compute core statistics
    stats = {
        "total_findings": prepared_data["FindingsNumber"].sum(),
        "patients_with_findings_list": combined_findings,
        "patients_with_findings": combined_findings.sum(),
        "patients_without_findings": len(prepared_data) - combined_findings.sum(),
        "total_rejected_findings": prepared_data["NumberFindingsRejected"].sum(),
        "rejected_patient_ids": prepared_data[prepared_data["NumberFindingsRejected"] > 0]["PatientID"].tolist(),
        "findings_number_distribution": prepared_data["FindingsNumber"].value_counts().sort_index().to_dict(),
        "findings_count": prepared_data["FindingsNumber"].value_counts().sort_index()
    }

    # Calculate distribution statistics for Findings numbers
    stats["findings_stats"] = prepared_data["FindingsNumber"].dropna().describe()

    # Calculate percentages for findings numbers
    total_patients = len(prepared_data)
    stats["findings_percentages"] = {
        category: round((count / total_patients) * 100, 2)
        for category, count in stats["findings_number_distribution"].items()
    }

    # Integrate findings distribution and percentages into a DataFrame for easier logging
    stats["findings_distribution_table"] = pd.DataFrame.from_dict({
        "Findings Count": stats["findings_number_distribution"].keys(),
        "Count": stats["findings_number_distribution"].values(),
        "Percentage": [stats["findings_percentages"].get(category, 0)
                       for category in stats["findings_number_distribution"].keys()]
    })

    return stats


def calculate_patient_findings_source(prepared_data):
    """
    Calculate the number and percentage of patients categorized by findings source.

    Categories include:
        - AI Alone
        - Human + AI Together
        - Human Alone
        - Any AI (AI Alone or Human + AI Together)
        - Any Human (Human Alone or Human + AI Together)

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with findings data.

    Returns:
        stats (dict): Counts and percentages for each source category, including patient-specific lists for certain conditions.
    """
    # Initialize sets to track patients by source category
    sources = {
        "AI Alone": set(),
        "Human + AI Together": set(),
        "Human Alone": set(),
        "Any AI": set(),
        "Any Human": set()
    }

    # Lists to indicate presence of specific categories for each patient
    has_human_ai_together = []
    has_human_alone = []
    has_ai_alone = []

    # Iterate through each patient to categorize their findings
    for idx, row in prepared_data.iterrows():
        patient_sources = set()

        for i in range(1, int(row["FindingsNumber"]) + 1):
            source_col = f"FindingSource{i}"
            source = row[source_col]
            if pd.notna(source):
                patient_sources.add(source)

        # Assign patient to specific categories based on sources
        if patient_sources == {"AI Alone"}:
            sources["AI Alone"].add(row["PatientID"])
        elif patient_sources == {"Human Alone"}:
            sources["Human Alone"].add(row["PatientID"])
        elif "Human + AI Together" in patient_sources:
            sources["Human + AI Together"].add(row["PatientID"])

        # Assign patient to broader categories
        if any(src in patient_sources for src in ["AI Alone", "Human + AI Together"]):
            sources["Any AI"].add(row["PatientID"])
        if any(src in patient_sources for src in ["Human Alone", "Human + AI Together"]):
            sources["Any Human"].add(row["PatientID"])

        # Populate the lists for specific conditions
        has_human_ai_together.append("Human + AI Together" in patient_sources)
        has_human_alone.append(
            "Human Alone" in patient_sources and "Human + AI Together" not in patient_sources)
        has_ai_alone.append(
            "AI Alone" in patient_sources and "Human + AI Together" not in patient_sources)

    # Calculate counts and percentages
    total_patients = len(prepared_data)
    stats = {}
    for source, patient_set in sources.items():
        count = len(patient_set)
        percentage = round((count / total_patients) *
                           100, 2) if total_patients > 0 else 0
        stats[source] = {"count": count, "percentage": percentage}

    # Add the new lists to the output stats
    stats["patient_specific_lists"] = {
        "has_human_ai_together": has_human_ai_together,
        "has_human_alone": has_human_alone,
        "has_ai_alone": has_ai_alone}

    return stats


def calculate_lesion_findings_source(prepared_data):
    """
    Calculate the number and percentage of findings (lesions) by source.

    Categories include:
        - AI Alone
        - Human + AI Together
        - Human Alone
        - Any AI (AI Alone or Human + AI Together)
        - Any Human (Human Alone or Human + AI Together)

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with findings data.

    Returns:
        stats (dict): Counts and percentages for each source category.
    """
    # Initialize counts for each source category
    sources = {
        "AI Alone": 0,
        "Human + AI Together": 0,
        "Human Alone": 0,
        "Any AI": 0,
        "Any Human": 0
    }

    # Iterate through each lesion to categorize by source
    for idx, row in prepared_data.iterrows():
        for i in range(1, int(row["FindingsNumber"]) + 1):
            source_col = f"FindingSource{i}"
            source = row[source_col]
            if pd.notna(source):
                if source == "AI Alone":
                    sources["AI Alone"] += 1
                elif source == "Human + AI Together":
                    sources["Human + AI Together"] += 1
                elif source == "Human Alone":
                    sources["Human Alone"] += 1

                if source in ["AI Alone", "Human + AI Together"]:
                    sources["Any AI"] += 1
                if source in ["Human Alone", "Human + AI Together"]:
                    sources["Any Human"] += 1

    # Calculate percentages for each source category
    total_findings = prepared_data["FindingsNumber"].sum()
    stats = {}
    for source, count in sources.items():
        percentage = round((count / total_findings) *
                           100, 2) if total_findings > 0 else 0
        stats[source] = {"count": count, "percentage": percentage}

    return stats


def analyze_radiologist_evaluation(prepared_data):
    """
    Analyze radiologist evaluation to determine PI-RADS scores distribution, percentages,
    and patient findings distribution based on "Human Alone" or "Human + AI Together" sources.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with radiologist evaluation data.

    Returns:
        stats (dict): Summary statistics including:
            - Highest PI-RADS scores distribution and percentages (Any Human)
            - Patient findings distribution and statistics (mean, std, min, max, median, Q1, Q3, IQR).
            - Distribution for all PI-RADS scores (aggregated)
            - Highest PI-RADS scores distribution for Human Alone findings
            - Distribution for all PI-RADS scores for Human Alone findings
    """
    # Ensure relevant columns are numeric or valid
    for col in ["PIRADS", "PIRADS2", "PIRADS3", "PIRADS4"]:
        prepared_data[col] = pd.to_numeric(prepared_data[col], errors="coerce")

    # Determine the highest PI-RADS score per patient
    prepared_data["highest_PIRADS_Score"] = prepared_data[[
        "PIRADS", "PIRADS2", "PIRADS3", "PIRADS4"]].max(axis=1)

    # Set PI-RADS score to "<3" if all PIRADS columns are NaN
    prepared_data["highest_PIRADS_Score"] = prepared_data["highest_PIRADS_Score"].fillna(
        0)

    # Highest PI-RADS Scores Distribution
    highest_pirads_distribution = prepared_data["highest_PIRADS_Score"].value_counts(
    ).sort_index().to_dict()
    total_highest_pirads_scores = prepared_data["highest_PIRADS_Score"].count()
    highest_pirads_percentages = {
        score: round((count / total_highest_pirads_scores) * 100, 2)
        for score, count in highest_pirads_distribution.items()
    }

    # Create a DataFrame for Highest PI-RADS distribution and percentages
    highest_pirads_table = pd.DataFrame.from_dict({
        "Score": highest_pirads_distribution.keys(),
        "Count": highest_pirads_distribution.values(),
        "Percentage": [highest_pirads_percentages.get(score, 0) for score in highest_pirads_distribution.keys()]
    })

    # Aggregate all PIRADS scores across PIRADS columns to get lesion-wise distribution
    # Flatten PIRADS values and drop NaN
    all_pirads_scores = prepared_data[[
        "PIRADS", "PIRADS2", "PIRADS3", "PIRADS4"]].values.flatten()
    all_pirads_scores = pd.Series(all_pirads_scores).dropna()

    # Calculate the distribution and percentages
    all_pirads_distribution = all_pirads_scores.value_counts().sort_index().to_dict()
    total_all_pirads_scores = len(all_pirads_scores)
    all_pirads_percentages = {
        score: round((count / total_all_pirads_scores) * 100, 2)
        for score, count in all_pirads_distribution.items()
    }

    # Create a DataFrame for all PI-RADS distribution and percentages
    all_pirads_table = pd.DataFrame.from_dict({
        "Score": all_pirads_distribution.keys(),
        "Count": all_pirads_distribution.values(),
        "Percentage": [all_pirads_percentages.get(score, 0) for score in all_pirads_distribution.keys()]
    })

    # Process all FindingSource columns
    finding_source_columns = [
        col for col in prepared_data.columns if col.startswith("FindingSource")]

    # Filter patients with findings from "Human Alone" or "Human + AI Together"
    human_findings = prepared_data[finding_source_columns].apply(
        lambda row: row.isin(["Human Alone", "Human + AI Together"]).sum(), axis=1)

    # Patient findings distribution (number of patients with 0, 1, 2, ... findings)
    findings_distribution = human_findings.value_counts().sort_index().to_dict()
    total_patients = len(human_findings)
    findings_percentages = {
        count: round((num / total_patients) * 100, 2)
        for count, num in findings_distribution.items()
    }

    # Create a DataFrame for all findings distribution and percentages
    findings_table = pd.DataFrame.from_dict({
        "Findings Nr.": findings_distribution.keys(),
        "Number of patients count": findings_distribution.values(),
        "Percentage": [findings_percentages.get(score, 0) for score in findings_distribution.keys()]
    })

    # Statistics for the number of findings
    findings_stats = human_findings.describe()

    # Human Alone
    # Patients with only "Human Alone" findings
    human_alone_patients = prepared_data[
        prepared_data[finding_source_columns].apply(
            lambda row: set(row.dropna()) == {"Human Alone"}, axis=1)
    ]

    # Highest PI-RADS scores for "Human Alone" patients
    human_alone_highest_pirads = human_alone_patients["highest_PIRADS_Score"].value_counts(
    ).sort_index().to_dict()

    # All PI-RADS scores for "Human Alone" patients
    human_alone_pirads_scores = human_alone_patients[[
        "PIRADS", "PIRADS2", "PIRADS3", "PIRADS4"]].values.flatten()
    human_alone_pirads_scores = pd.Series(human_alone_pirads_scores).dropna()
    human_alone_pirads_distribution = human_alone_pirads_scores.value_counts(
    ).sort_index().to_dict()

    # Compile results
    stats = {
        "highest_PIRADS_Score": prepared_data["highest_PIRADS_Score"].tolist(),
        "highest_pirads_distribution": highest_pirads_distribution,
        "highest_pirads_percentages": highest_pirads_percentages,
        "highest_pirads_table": highest_pirads_table,
        "all_pirads_distribution": all_pirads_distribution,
        "all_pirads_percentages": all_pirads_percentages,
        "all_pirads_table": all_pirads_table,
        "findings_distribution": findings_distribution,
        "findings_distribution_list": list(findings_distribution.values()),
        "findings_percentages": findings_percentages,
        "findings_table": findings_table,
        "findings_stats": findings_stats,
        "human_alone_highest_pirads": human_alone_highest_pirads,
        "human_alone_pirads_distribution": human_alone_pirads_distribution
    }

    return stats


def analyze_ai_evaluation(prepared_data):
    """
    Analyze AI evaluation to determine PI-RADS scores distribution, percentages,
    and patient findings distribution based on "AI Alone" or "Human + AI Together" sources.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with AI score columns.

    Returns:
        stats (dict): Summary statistics for AI scores, highest scores per patient, and frequancy of findings.
    """
    # Identify columns containing AI scores
    ai_columns = [
        col for col in prepared_data.columns if col.startswith("AIScore")
    ]

    # Convert AI score columns to numeric, handling errors
    prepared_data[ai_columns] = prepared_data[ai_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Combine all AI scores into a single series, dropping missing values
    all_scores = prepared_data[ai_columns].stack().dropna()

    # Calculate the highest AI score per patient
    highest_scores = prepared_data[ai_columns].max(axis=1)

    # Process all FindingSource columns
    finding_source_columns = [
        col for col in prepared_data.columns if col.startswith("FindingSource")]

    # Filter patients with findings from "AI Alone" or "Human + AI Together"
    ai_findings = prepared_data[finding_source_columns].apply(
        lambda row: row.isin(["AI Alone", "Human + AI Together"]).sum(), axis=1)

    # Patient findings distribution (number of patients with 0, 1, 2, ... findings)
    findings_distribution = ai_findings.value_counts().sort_index().to_dict()
    total_patients = len(ai_findings)
    findings_percentages = {
        count: round((num / total_patients) * 100, 2)
        for count, num in findings_distribution.items()
    }

    # Create a DataFrame for all findings number distribution and percentages
    findings_table = pd.DataFrame.from_dict({
        "Findings Nr.": findings_distribution.keys(),
        "Number of patients count": findings_distribution.values(),
        "Percentage": [findings_percentages.get(score, 0) for score in findings_distribution.keys()]
    })

    # Statistics for the number of findings
    findings_stats = ai_findings.describe()

    # Generate descriptive statistics for all scores and highest scores
    stats = {
        "all_scores_stats": all_scores.describe(),
        "highest_scores_stats": highest_scores.describe(),
        "ai_score_distribution": highest_scores.value_counts().sort_index(),
        "highest_scores": highest_scores.fillna(0),
        "findings_distribution": findings_distribution,
        "findings_count": ai_findings,
        "findings_percentages": findings_percentages,
        "findings_table": findings_table,
        "findings_stats": findings_stats
    }
    return stats


def analyze_biopsy_statistics(prepared_data):
    """
    Analyze biopsy statistics including targeted and systematic biopsies.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with biopsy data.

    Returns:
        stats (dict): Summary statistics for biopsy data.
    """
    prepared_data["NumberofTargetedBiopsies"] = pd.to_numeric(
        prepared_data["NumberofTargetedBiopsies"], errors="coerce").fillna(0).astype(int)
    prepared_data["NumberofSystematicBiopsies"] = pd.to_numeric(
        prepared_data["NumberofSystematicBiopsies"], errors="coerce").fillna(0).astype(int)

    # Counts for biopsy types
    stats = {
        "all_biopsies_count": ((prepared_data["NumberofTargetedBiopsies"] > 0) |
                               (prepared_data["NumberofSystematicBiopsies"] > 0)).sum(),
        "any_targeted_biopsy_count": (prepared_data["NumberofTargetedBiopsies"] > 0).sum(),
        "any_systematic_biopsy_count": (prepared_data["NumberofSystematicBiopsies"] > 0).sum(),
        "both_biopsies_count": ((prepared_data["NumberofTargetedBiopsies"] > 0) &
                                (prepared_data["NumberofSystematicBiopsies"] > 0)).sum(),
        "only_targeted_biopsy_count": ((prepared_data["NumberofTargetedBiopsies"] > 0) &
                                       (prepared_data["NumberofSystematicBiopsies"] == 0)).sum(),
        "only_systematic_biopsy_count": ((prepared_data["NumberofSystematicBiopsies"] > 0) &
                                         (prepared_data["NumberofTargetedBiopsies"] == 0)).sum()
    }

    # Distribution of biopsies and stats
    stats["targeted_biopsy_distribution"] = prepared_data["NumberofTargetedBiopsies"].value_counts(
    ).sort_index()
    stats["targeted_biopsy_distribution_stats"] = prepared_data["NumberofTargetedBiopsies"].value_counts(
    ).sort_index().describe()
    stats["systematic_biopsy_distribution"] = prepared_data["NumberofSystematicBiopsies"].value_counts(
    ).sort_index()
    stats["systematic_biopsy_distribution_stats"] = prepared_data["NumberofSystematicBiopsies"].value_counts(
    ).sort_index().describe()

    # Add biopsy category for each patient (0: none, 1: both, 2: only targeted, 3: only systimatic)
    stats["biopsy_method_list"] = prepared_data.apply(
        lambda row: 0 if (row["NumberofTargetedBiopsies"] == 0 and row["NumberofSystematicBiopsies"] == 0) else
        1 if (row["NumberofTargetedBiopsies"] > 0 and row["NumberofSystematicBiopsies"] > 0) else
        2 if (row["NumberofTargetedBiopsies"] > 0 and row["NumberofSystematicBiopsies"] == 0) else
        3 if (row["NumberofSystematicBiopsies"] > 0 and row["NumberofTargetedBiopsies"] == 0) else
        None,
        axis=1
    ).tolist()

    return stats


def analyze_ggg_values(prepared_data):
    """
    Analyze GGG values for different conditions including overall highest, all lesions, and filtered 
    human/AI findings.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe containing GGG and FindingSource columns.

    Returns:
        stats (dict): A dictionary containing GGG distributions and percentages for various scenarios.
    """
    # Helper function: Clean GGG values
    def clean_ggg(value):
        if isinstance(value, str) and "0 (<1)" in value:
            return 0
        try:
            return pd.to_numeric(value, errors="coerce")
        except:
            return np.nan

    # Helper function: Calculate GGG distribution and percentages
    def calculate_distribution(series):
        series = series.dropna()
        distribution = series.value_counts().sort_index()
        total = len(series)
        percentages = (distribution / total * 100).round(2)
        # Calculate 95% Confidence Intervals
        ci_bounds = [
            calculate_confidence_interval(percentage, total_cases=total)
            for percentage in percentages]
        ci_lower, ci_upper = zip(*ci_bounds)
        return pd.DataFrame({
            "Count": distribution,
            "Percentage": percentages,
            "95CI Lower": ci_lower,
            "95CI Upper": ci_upper
        }).reset_index().rename(columns={"index": "GGG Value"})

    # Helper function: Extract relevant GGG columns
    def extract_ggg_columns(df):
        return [col for col in df.columns if "GGG" in col]

    ggg_columns = extract_ggg_columns(prepared_data)

    # 1. Overall Highest GGG Value per Patient
    prepared_data["highest_GGG_overall"] = prepared_data[ggg_columns].applymap(
        clean_ggg).max(axis=1)
    overall_highest_distribution = calculate_distribution(
        prepared_data["highest_GGG_overall"])

    # 2. All Lesions GGG Distribution (aggregate all values)
    all_ggg_values = prepared_data[ggg_columns].applymap(clean_ggg).stack()
    all_lesions_distribution = calculate_distribution(all_ggg_values)

    # 3. Highest GGG for Patients with Human Findings
    finding_source_columns = [
        col for col in prepared_data.columns if col.startswith("FindingSource")]
    human_rows = prepared_data[finding_source_columns].apply(
        lambda row: any(val in ["Human Alone", "Human + AI Together"] for val in row.dropna()), axis=1
    )
    highest_ggg_human = prepared_data.loc[human_rows, ggg_columns].applymap(
        clean_ggg).max(axis=1)
    human_highest_distribution = calculate_distribution(highest_ggg_human)

    # 4. Highest GGG for Patients with AI Findings
    ai_rows = prepared_data[finding_source_columns].apply(
        lambda row: any(val in ["AI Alone", "Human + AI Together"] for val in row.dropna()), axis=1
    )
    highest_ggg_ai = prepared_data.loc[ai_rows, ggg_columns].applymap(
        clean_ggg).max(axis=1)
    ai_highest_distribution = calculate_distribution(highest_ggg_ai)

    # Helper function: Extract GGG values for specific findings source conditions.
    def extract_ggg_for_findings(prepared_data, finding_source_values):
        finding_source_columns = [
            col for col in prepared_data.columns if col.startswith("FindingSource")]
        ggg_values = []

        # Iterate through rows to check FindingSource and collect GGG values
        for _, row in prepared_data.iterrows():
            for col in finding_source_columns:
                if pd.notna(row[col]) and row[col] in finding_source_values:
                    index = col.replace("FindingSource", "")
                    ggg_col = f"LesionOverallGGGTargeted{index}"
                    if ggg_col in prepared_data.columns and pd.notna(row[ggg_col]):
                        ggg_values.append(clean_ggg(row[ggg_col]))

        return pd.Series(ggg_values)

    # 5. All Lesions GGG for Human Findings
    human_ggg_values = extract_ggg_for_findings(
        prepared_data, ["Human Alone", "Human + AI Together"])
    human_all_lesions_distribution = calculate_distribution(human_ggg_values)

    # 6. All Lesions GGG for AI Findings
    ai_ggg_values = extract_ggg_for_findings(
        prepared_data, ["AI Alone", "Human + AI Together"])
    ai_all_lesions_distribution = calculate_distribution(ai_ggg_values)

    # 7. Get list of significant cancer or not
    # Significant Cancer Detection: 1 if GGG > 1, otherwise 0

    prepared_data["highest_GGG_overall_all"] = prepared_data[ggg_columns].applymap(
        clean_ggg).max(axis=1)
    prepared_data["highest_GGG_overall_all"].fillna(
        0, inplace=True)  # NaN rows get 0

    # Add columns for significant and non-significant cancer detection, patient-level
    patient_has_cspca = prepared_data["highest_GGG_overall_all"].apply(
        lambda x: 1 if x > 1 else 0)
    patient_has_non_cspca = prepared_data["highest_GGG_overall_all"].apply(
        lambda x: 1 if x < 2 else 0)

    # Add columns for significant and non-significant cancer detection, lesion-level
    ai_lesion_has_cspca = ai_ggg_values.apply(lambda x: 1 if x > 1 else 0)
    ai_lesion_has_non_cspca = ai_ggg_values.apply(lambda x: 1 if x < 2 else 0)

    # 8. Aggregate all lesions and significant lesions based on finding sources
    significant_human_lesions = []
    non_significant_human_lesions = []
    significant_ai_lesions = []
    non_significant_ai_lesions = []

    for _, row in prepared_data.iterrows():
        for col in finding_source_columns:
            if pd.notna(row[col]):
                index = col.replace("FindingSource", "")
                ggg_col = f"LesionOverallGGGTargeted{index}"
                if ggg_col in prepared_data.columns and pd.notna(row[ggg_col]):
                    ggg_value = clean_ggg(row[ggg_col])
                    if ggg_value > 1:
                        if row[col] in ["Human Alone", "Human + AI Together"]:
                            significant_human_lesions.append(1)
                        else:
                            significant_human_lesions.append(0)
                        if row[col] in ["AI Alone", "Human + AI Together"]:
                            significant_ai_lesions.append(1)
                        else:
                            significant_ai_lesions.append(0)
                    else:
                        if row[col] in ["Human Alone", "Human + AI Together"]:
                            non_significant_human_lesions.append(1)
                        else:
                            non_significant_human_lesions.append(0)
                        if row[col] in ["AI Alone", "Human + AI Together"]:
                            non_significant_ai_lesions.append(1)
                        else:
                            non_significant_ai_lesions.append(0)

    # 9. Aggregate patient-level significant and non-significant counts using all ggg_columns

    significant_human_patients = prepared_data.apply(
        lambda row: (
            1 if any(
                row[col] in ["Human Alone",
                             "Human + AI Together"] and clean_ggg(row[ggg_col]) > 1
                for col in finding_source_columns for ggg_col in ggg_columns
            ) else (
                0 if any(
                    row[col] in ["Human Alone",
                                 "Human + AI Together"] and clean_ggg(row[ggg_col]) < 2
                    for col in finding_source_columns for ggg_col in ggg_columns
                ) else None
            )
        ), axis=1
    ).dropna().tolist()

    non_significant_human_patients = [
        1 if value == 0 else 0 for value in significant_human_patients]

    significant_ai_patients = prepared_data.apply(
        lambda row: (
            1 if any(
                row[col] in ["AI Alone",
                             "Human + AI Together"] and clean_ggg(row[ggg_col]) > 1
                for col in finding_source_columns for ggg_col in ggg_columns
            ) else (
                0 if any(
                    row[col] in ["AI Alone",
                                 "Human + AI Together"] and clean_ggg(row[ggg_col]) < 2
                    for col in finding_source_columns for ggg_col in ggg_columns
                ) else None
            )
        ), axis=1
    ).dropna().tolist()

    non_significant_ai_patients = [
        1 if value == 0 else 0 for value in significant_ai_patients]
    # Combine stats
    stats = {
        "overall_highest_GGG": overall_highest_distribution,
        "all_lesions_GGG": all_lesions_distribution,
        "highest_GGG_human_findings": human_highest_distribution,
        "highest_GGG_ai_findings": ai_highest_distribution,
        "all_lesions_GGG_human_findings": human_all_lesions_distribution,
        "all_lesions_GGG_ai_findings": ai_all_lesions_distribution,
        "patient_has_overall_cspca": patient_has_cspca.tolist(),
        "patient_has_overall_non_cspca": patient_has_non_cspca.tolist(),
        "overall_highest_GGG_list": prepared_data["highest_GGG_overall"].fillna(0),
        "ai_lesion_has_cspca": ai_lesion_has_cspca.tolist(),
        "ai_lesion_has_non_cspca": ai_lesion_has_non_cspca.tolist(),
        "significant_human_lesions": significant_human_lesions,
        "non_significant_human_lesions": non_significant_human_lesions,
        "significant_ai_lesions": significant_ai_lesions,
        "non_significant_ai_lesions": non_significant_ai_lesions,
        "significant_human_patients": significant_human_patients,
        "non_significant_human_patients": non_significant_human_patients,
        "significant_ai_patients": significant_ai_patients,
        "non_significant_ai_patients": non_significant_ai_patients
    }

    # 10. Calculate significant non-significant counts and percentages and 95% confidence intervals
    # Metrics to analyze
    metrics = {
        "patient_has_cspca": patient_has_cspca,
        "patient_has_non_cspca": patient_has_non_cspca,
        "significant_ai_patients": significant_ai_patients,
        "non_significant_ai_patients": non_significant_ai_patients,
        "significant_human_patients": significant_human_patients,
        "non_significant_human_patients": non_significant_human_patients,
        "significant_ai_lesions": significant_ai_lesions,
        "non_significant_ai_lesions": non_significant_ai_lesions,
        "significant_human_lesions": significant_human_lesions,
        "non_significant_human_lesions": non_significant_human_lesions
    }
    # Compute counts, percentages, and CIs

    def compute_stats_and_cis(metric_list, total_count):
        """Compute count, percentage, and 95% CI for a metric."""
        count = sum(metric_list)
        percentage = (count / total_count * 100) if total_count > 0 else 0
        ci = calculate_confidence_interval(percentage, total_count)
        return {"count": count, "percentage": percentage, "95CI": ci}

    # Total patient count and lesion count
    total_patients = len(prepared_data)
    total_human_patients = len(significant_human_patients)
    total_ai_patients = len(significant_ai_patients)
    total_human_lesions = sum(significant_human_lesions) + \
        sum(non_significant_human_lesions)
    total_ai_lesions = sum(significant_ai_lesions) + \
        sum(non_significant_ai_lesions)

    # Add stats for each metric
    for metric_name, metric_values in metrics.items():
        if "patient_has" in metric_name:
            total_count = total_patients
        elif "human_patients" in metric_name:
            total_count = total_human_patients
        elif "ai_patients" in metric_name:
            total_count = total_ai_patients
        elif "human_lesions" in metric_name:
            total_count = total_human_lesions
        elif "ai_lesions" in metric_name:
            total_count = total_ai_lesions
        stats[f"{metric_name}_stats"] = compute_stats_and_cis(
            metric_values, total_count)

    return stats


def generate_findings_statistics(prepared_data, log_file="findings_statistics.log"):
    """
    Main function to generate findings statistics and log results.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe containing all data.
        log_file (str): File path for logging results.

    Returns:
        results (dict): Consolidated results from all analyses.
    """
    # Process each section
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(log_file, "Timestamp", current_time)
    total_cases = len(prepared_data)
    log_to_file(log_file, "Total number of cases",
                f"Total number of cases analyzed: {total_cases}\n")

    # Combined findings
    combined_stats = analyze_combined_findings(prepared_data)
    combined_log = (
        f"Total Findings: {combined_stats['total_findings']}\n"
        f"Patients with Findings: {combined_stats['patients_with_findings']}\n"
        f"Patients without Findings: {combined_stats['patients_without_findings']}\n"
        f"Total Rejected Findings: {combined_stats['total_rejected_findings']}\n"
        f"Number of Patients with Rejected Findings: {len(combined_stats['rejected_patient_ids'])}\n"
        f"Patient with Rejected Findings IDs: {', '.join(combined_stats['rejected_patient_ids'])}\n"
        f"Findings Number Distribution:\n"
    )
    combined_log += combined_stats["findings_distribution_table"].to_string(
        index=False)
    combined_log += "\nCombined finding numbers statistics:\n"
    combined_log += combined_stats["findings_stats"].to_string()
    log_to_file(log_file, "Combined Findings", combined_log)

    # Patient Findings Source
    patient_findings_source_stats = calculate_patient_findings_source(
        prepared_data)
    # Convert to DataFrame and transpose
    patient_findings_df = pd.DataFrame(patient_findings_source_stats).T
    log_to_file(log_file, "Patient Findings Source",
                patient_findings_df.to_string(index=True))

    # Lesion Findings by Source
    lesion_findings_source_stats = calculate_lesion_findings_source(
        prepared_data)
    # Convert to DataFrame and transpose
    lesion_findings_df = pd.DataFrame(lesion_findings_source_stats).T
    log_to_file(log_file, "Lesion Findings by Source",
                lesion_findings_df.to_string(index=True))

    # Radiologist Evaluation
    radiologist_stats = analyze_radiologist_evaluation(prepared_data)
    radiologist_log = "Highest PI-RADS Scores Distribution:\n"
    radiologist_log += radiologist_stats["highest_pirads_table"].to_string(
        index=False)
    radiologist_log += "\n\nAll PI-RADS Scores Distribution:\n"
    radiologist_log += radiologist_stats["all_pirads_table"].to_string(
        index=False)
    radiologist_log += "\n\nPatient with number of Findings Distribution:\n"
    radiologist_log += radiologist_stats["findings_table"].to_string(
        index=False)
    radiologist_log += "\nRadiologist finding numbers statistics:\n"
    radiologist_log += radiologist_stats["findings_stats"].to_string()
    radiologist_log += "\n<< Human Alone Findings Analysis >>\n"
    radiologist_log += "\nHuman Alone Highest PI-RADS Scores Distribution:\n"
    radiologist_log += pd.Series(
        radiologist_stats["human_alone_highest_pirads"]).to_string()
    radiologist_log += "\nHuman All PI-RADS Scores Distribution:\n"
    radiologist_log += pd.Series(
        radiologist_stats["human_alone_pirads_distribution"]).to_string()
    log_to_file(log_file, "Radiologist Evaluation", radiologist_log)

    # AI score analysis
    ai_stats = analyze_ai_evaluation(prepared_data)
    ai_log = "Highest AI Scores Stats:\n"
    ai_log += ai_stats["highest_scores_stats"].to_string()
    ai_log += "\n\nAll AI Scores Stats:\n"
    ai_log += ai_stats["all_scores_stats"].to_string()
    ai_log += "\n\nPatient with number of Findings Distribution:\n"
    ai_log += ai_stats["findings_table"].to_string(
        index=False)
    ai_log += "\nAI finding numbers statistics:\n"
    ai_log += ai_stats["findings_stats"].to_string()
    log_to_file(log_file, "AI Evaluation", ai_log)

    # Biopsy statistics
    biopsy_stats = analyze_biopsy_statistics(prepared_data)
    biopsy_log = "Biopsy methods count:\n"
    biopsy_log += (
        f"Patients with any biopsy type count: {biopsy_stats['all_biopsies_count']}\n"
        f"Patients with any Targeted biopsy count: {biopsy_stats['any_targeted_biopsy_count']}\n"
        f"Patients with any Systematic biopsy counts: {biopsy_stats['any_systematic_biopsy_count']}\n"
        f"Patients with Both Targeted and Systematic biopsy count: {biopsy_stats['both_biopsies_count']}\n"
        f"Patients with Only Targeted biopsy count: {biopsy_stats['only_targeted_biopsy_count']}\n"
        f"Patients with Only Systematic biopsy counts: {biopsy_stats['only_systematic_biopsy_count']}\n"
    )
    biopsy_log += "\n\nTargeted biopsy Distribution:\n"
    biopsy_log += pd.Series(
        biopsy_stats["targeted_biopsy_distribution"]).to_string()
    biopsy_log += "\nTargeted biopsy distribution Stats:\n"
    biopsy_log += biopsy_stats['targeted_biopsy_distribution_stats'].to_string()
    biopsy_log += "\n\nSystematic biopsy Distribution:\n"
    biopsy_log += pd.Series(
        biopsy_stats["systematic_biopsy_distribution"]).to_string()
    biopsy_log += "\nSystematic biopsy distribution Stats:\n"
    biopsy_log += biopsy_stats['systematic_biopsy_distribution_stats'].to_string()
    log_to_file(log_file, "Biopsy Statistics", biopsy_log)

    # GGG statistics
    ggg_stats = analyze_ggg_values(prepared_data)
    ggg_log = "\n\n<<< GGG Values Analysis >>>\n"
    # Log Overall Highest GGG
    ggg_log += "\nOverall Highest GGG Value Distribution:\n"
    ggg_log += ggg_stats["overall_highest_GGG"].to_string(index=False)
    # Log All Lesions GGG
    ggg_log += "\n\nAll Lesions GGG Value Distribution:\n"
    ggg_log += ggg_stats["all_lesions_GGG"].to_string(index=False)
    # Log Human Findings GGG
    ggg_log += "\n\nHighest GGG Value for Human Findings:\n"
    ggg_log += ggg_stats["highest_GGG_human_findings"].to_string(index=False)
    ggg_log += "\n\nAll Lesions GGG for Human Findings:\n"
    ggg_log += ggg_stats["all_lesions_GGG_human_findings"].to_string(
        index=False)
    # Log AI Findings GGG
    ggg_log += "\n\nHighest GGG Value for AI Findings:\n"
    ggg_log += ggg_stats["highest_GGG_ai_findings"].to_string(index=False)
    ggg_log += "\n\nAll Lesions GGG for AI Findings:\n"
    ggg_log += ggg_stats["all_lesions_GGG_ai_findings"].to_string(index=False)
    # Significant cancer
    ggg_log += "\n\nSignificant cancer:\n"
    for metric_name, metric_stats in ggg_stats.items():
        if metric_name.endswith("_stats"):
            ggg_log += (
                f"\nMetric: {metric_name.replace('_stats', '').replace('_', ' ').capitalize()}\n"
                f"  Count: {metric_stats['count']}\n"
                f"  Percentage: {metric_stats['percentage']:.2f}%\n"
                f"  95% Confidence Interval: ({metric_stats['95CI'][0]:.2f}%, {metric_stats['95CI'][1]:.2f}%)\n"
            )

    # Write to Log File
    log_to_file(log_file, "GGG Values Analysis", ggg_log)

    # Combine results into a single dictionary
    results = {
        "combined_stats": combined_stats,
        "patient_findings_source_stats": patient_findings_source_stats,
        "lesion_findings_source_stats": lesion_findings_source_stats,
        "radiologist_stats": radiologist_stats,
        "ai_stats": ai_stats,
        "biopsy_stats": biopsy_stats,
        "ggg_stats": ggg_stats
    }

    return results
