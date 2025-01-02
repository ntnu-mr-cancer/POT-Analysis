"""
Characteristics statistics code, part of the
Analysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""

import numpy as np
import pandas as pd
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


def calculate_baseline_stats(prepared_data):
    """
    Calculate general descriptive statistics for and add derived metrics.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe.

    Returns:
        stats (pd.DataFrame): Summary statistics for columns.
    """
    numeric_columns = ["Age", "Height", "Weight",
                       "PSAlevel", "ProstateGlandVolume"]

    # Normalize and convert Height values
    prepared_data["Height"] = prepared_data["Height"].apply(
        lambda x: float(str(x).replace(",", ".").strip()) / 100
        if float(str(x).replace(",", ".").strip()) > 10
        else float(str(x).replace(",", ".").strip())
    )

    # Convert numeric columns to float
    prepared_data[numeric_columns] = prepared_data[numeric_columns].apply(
        lambda col: col.astype(str).str.replace(",", ".").astype(float))

    # Add derived column PSADensity
    prepared_data["PSADensity"] = prepared_data["PSAlevel"] / \
        prepared_data["ProstateGlandVolume"]

    # Calculate descriptive statistics
    stats = prepared_data[numeric_columns + ["PSADensity"]].describe().T
    stats["median"] = prepared_data[numeric_columns + ["PSADensity"]].median()
    stats.insert(0, "Units", ["Years", "Meters",
                 "Kg", "ng/mL", "mL", "ng/mL²"])
    return stats


def calculate_previous_statistics(prepared_data):
    """
    Calculate the counts and percentages for DRE findings and Family History categories.

    Parameters:
        prepared_data (pd.DataFrame): Input DataFrame containing 'DREfinding' and 'FamilyHistory'.

    Returns:
        dre_table (pd.DataFrame): Table showing counts and percentages for 'DREfinding'.
        family_history_table (pd.DataFrame): Table showing counts and percentages for 'FamilyHistory'.
    """
    # Calculate counts and percentages for DRE
    dre_counts = prepared_data['DREfinding'].value_counts()
    dre_percentages = prepared_data['DREfinding'].value_counts(
        normalize=True) * 100

    # Combine into a DataFrame
    dre_table = pd.DataFrame({
        'Count': dre_counts,
        'Percentage': dre_percentages
    }).reset_index()
    dre_table.rename(columns={'index': 'DREfinding'}, inplace=True)

    # Calculate counts and percentages for Family History
    family_history_counts = prepared_data['FamilyHistory'].value_counts()
    family_history_percentages = prepared_data['FamilyHistory'].value_counts(
        normalize=True) * 100

    # Combine into a DataFrame
    family_history_table = pd.DataFrame({
        'Count': family_history_counts,
        'Percentage': family_history_percentages
    }).reset_index()
    family_history_table.rename(
        columns={'index': 'FamilyHistory'}, inplace=True)

    return dre_table, family_history_table


def analyze_treatment(prepared_data):
    """
    Analyze treatment data to identify patterns and completeness.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with treatment data.

    Returns:
        treatment_table (pd.DataFrame): Table showing counts and percentages for treated/untreated patients.
        treatment_types_table (pd.DataFrame): Table showing counts and percentages for treatment type for those who trated.
    """
    # Fill missing treatment type for treated patients with "Unknown"
    prepared_data.loc[
        (prepared_data["HavePatientTreated"] == "Yes") & (
            prepared_data["TreatmentType"].isna()),
        "TreatmentType"
    ] = "Unknown"

    # Count and percentage for treated/untreated patients
    treatment_counts = prepared_data["HavePatientTreated"].value_counts()
    treatment_percentages = prepared_data["HavePatientTreated"].value_counts(
        normalize=True) * 100

    treatment_table = pd.DataFrame({
        'Count': treatment_counts,
        'Percentage': treatment_percentages
    }).reset_index()
    treatment_table.rename(columns={'index': 'TreatmentStatus'}, inplace=True)

    # Count and percentage of treatment types for treated patients
    treatment_types_counts = prepared_data[prepared_data["HavePatientTreated"]
                                           == "Yes"]["TreatmentType"].value_counts()
    treatment_types_percentages = prepared_data[prepared_data["HavePatientTreated"]
                                                == "Yes"]["TreatmentType"].value_counts(normalize=True) * 100

    treatment_types_table = pd.DataFrame({
        'Count': treatment_types_counts,
        'Percentage': treatment_types_percentages
    }).reset_index()
    treatment_types_table.rename(
        columns={'index': 'TreatmentType'}, inplace=True)

    return treatment_table, treatment_types_table


def process_dates(prepared_data):
    """
    Convert date columns to datetime and calculate date ranges and differences.

    Parameters:
        prepared_data (pd.DataFrame): The input dataframe with date columns.

    Returns:
        date_ranges (pd.DataFrame): Date ranges for each date column.
        stats (pd.DataFrame): Summary statistics for date differences between key pairs.
    """
    # List of date columns to process
    date_columns = [
        "RadiologistEvaluationDate", "BiopsyProcedureDate", "TreatmentDate",
        "CompleteDate", "HistopathologicalEvaluationDate", "ScanDate"
    ]

    # Convert date columns to datetime format, handling errors
    prepared_data[date_columns] = prepared_data[date_columns].apply(
        pd.to_datetime, format='%d.%m.%Y', errors='coerce'
    )

    # Calculate min and max date for each column
    date_ranges = {
        col: {"min": prepared_data[col].min(), "max": prepared_data[col].max()}
        for col in date_columns
    }
    date_ranges = pd.DataFrame(date_ranges)

    # Define pairs of dates to calculate differences
    date_pairs = {
        "ScanDate to RadiologistEvaluationDate": ("ScanDate", "RadiologistEvaluationDate"),
        "ScanDate to BiopsyProcedureDate": ("ScanDate", "BiopsyProcedureDate"),
        "ScanDate to HistopathologicalEvaluationDate": ("ScanDate", "HistopathologicalEvaluationDate"),
        "ScanDate to TreatmentDate": ("ScanDate", "TreatmentDate"),
        "ScanDate to CompleteDate": ("ScanDate", "CompleteDate"),
        "RadiologistEvaluationDate to BiopsyProcedureDate": ("RadiologistEvaluationDate", "BiopsyProcedureDate"),
        "RadiologistEvaluationDate to HistopathologicalEvaluationDate": ("RadiologistEvaluationDate", "HistopathologicalEvaluationDate"),
        "RadiologistEvaluationDate to TreatmentDate": ("RadiologistEvaluationDate", "TreatmentDate"),
        "RadiologistEvaluationDate to CompleteDate": ("RadiologistEvaluationDate", "CompleteDate"),
        "BiopsyProcedureDate to HistopathologicalEvaluationDate": ("BiopsyProcedureDate", "HistopathologicalEvaluationDate"),
        "BiopsyProcedureDate to TreatmentDate": ("BiopsyProcedureDate", "TreatmentDate"),
        "BiopsyProcedureDate to CompleteDate": ("BiopsyProcedureDate", "CompleteDate"),
        "HistopathologicalEvaluationDate to TreatmentDate": ("HistopathologicalEvaluationDate", "TreatmentDate"),
        "HistopathologicalEvaluationDate to CompleteDate": ("HistopathologicalEvaluationDate", "CompleteDate"),
        "TreatmentDate to CompleteDate": ("TreatmentDate", "CompleteDate")
    }

    date_diff_stats = []
    for label, (date1, date2) in date_pairs.items():
        # Filter rows with non-missing date values
        valid_dates = prepared_data[[date1, date2]].dropna()
        if valid_dates.empty:
            continue

        # Calculate differences in days between the date pairs
        diff_in_days = (valid_dates[date2] - valid_dates[date1]).dt.days

        # Compute summary statistics for the differences
        date_diff_stats.append({
            "Date Difference": label,
            "Count": diff_in_days.count(),
            "Mean": diff_in_days.mean(),
            "Min": diff_in_days.min(),
            "Max": diff_in_days.max(),
            "Median": diff_in_days.median(),
            "Std": diff_in_days.std(),
            "Q1": np.percentile(diff_in_days, 25),
            "Q3": np.percentile(diff_in_days, 75),
            "IQR": np.percentile(diff_in_days, 75) - np.percentile(diff_in_days, 25),
        })

    stats = pd.DataFrame(date_diff_stats)

    return date_ranges, stats


def generate_characteristics_statistics(prepared_data, log_file="characteristics_statistics.log"):
    """
    Main function to generate general participants characteristics statistics and log results.

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

    # Baseline participants statistics
    baseline_stats = calculate_baseline_stats(prepared_data)
    log_to_file(log_file, "Descriptive Statistics",
                baseline_stats.to_string())

    # Previous statistics
    dre_table, family_history_table = calculate_previous_statistics(
        prepared_data)
    log_to_file(log_file, "DRE Findings", dre_table.to_string())
    log_to_file(log_file, "Family history Findings",
                family_history_table.to_string())

    # Treatment analysis
    treatment_table, treatment_types = analyze_treatment(prepared_data)
    log_to_file(log_file, "Treatment Counts", treatment_table.to_string())
    log_to_file(log_file, "Treatment Types", treatment_types.to_string())

    # Date analysis
    date_ranges, date_differences_df = process_dates(prepared_data)
    log_to_file(log_file, "Date Ranges", date_ranges.to_string())
    log_to_file(log_file, "Date Differences (in days)",
                date_differences_df.to_string())

    # Combine results into a single dictionary
    results = {
        "baseline_stats": baseline_stats,
        "dre_stats": dre_table,
        "family_history_stats": family_history_table,
        "treatment_counts": treatment_table,
        "treatment_types": treatment_types,
        "date_ranges": date_ranges,
        "date_differences": date_differences_df
    }

    return results
