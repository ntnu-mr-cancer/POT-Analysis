"""
Feasability analysis code, part of the
Analysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""

from datetime import datetime
from scripts.utils import calculate_confidence_interval


def analyze_feasibility(prepared_data, log_file="feasibility_analysis.log"):
    """
    Analyzes the feasibility based on three variables:
    'SegmentationAcceptable', 'WasAlignmentAcceptable', 'HaveFacedTechnicalIssues'.
    Logs detailed results and calculates the feasibility status based on unacceptable cases.

    Parameters:
        prepared_data (pd.DataFrame): The cleaned and prepared data to analyze.
        log_file (str): The path to the log file where results will be written.

    Returns:
        feasibility_status (str): 'Feasible' if less than 10% of cases are unacceptable, otherwise 'Not Feasible'.
    """
    # Define acceptable conditions
    acceptable_conditions = {
        "SegmentationAcceptable": "Yes",
        "WasAlignmentAcceptable": "Yes",
        "HaveFacedTechnicalIssues": "No"
    }

    # Identify unacceptable cases
    unacceptable_cases = prepared_data[
        (prepared_data['SegmentationAcceptable'] != acceptable_conditions["SegmentationAcceptable"]) |
        (prepared_data['WasAlignmentAcceptable'] != acceptable_conditions["WasAlignmentAcceptable"]) |
        (prepared_data['HaveFacedTechnicalIssues'] !=
         acceptable_conditions["HaveFacedTechnicalIssues"])
    ]

    # Calculate summary metrics
    total_cases = len(prepared_data)
    num_unacceptable_cases = len(unacceptable_cases)
    overall_percentage = (num_unacceptable_cases /
                          total_cases) * 100 if total_cases > 0 else 0
    # Overall percentage CI
    overall_percentage_ci = calculate_confidence_interval(
        overall_percentage, total_cases)

    # Classify unacceptable cases for detailed analysis
    issue_counts = {
        "Segmentation": unacceptable_cases[unacceptable_cases['SegmentationAcceptable'] != acceptable_conditions["SegmentationAcceptable"]],
        "Alignment": unacceptable_cases[unacceptable_cases['WasAlignmentAcceptable'] != acceptable_conditions["WasAlignmentAcceptable"]],
        "Technical": unacceptable_cases[unacceptable_cases['HaveFacedTechnicalIssues'] != acceptable_conditions["HaveFacedTechnicalIssues"]]
    }

    # Calculate percentages for each criterion
    issue_percentages = {
        issue: (len(cases) / num_unacceptable_cases) *
        100 if num_unacceptable_cases > 0 else 0
        for issue, cases in issue_counts.items()}

    # Issue percentages CIs
    issue_percentages_ci = {
        issue: calculate_confidence_interval(
            percentage, num_unacceptable_cases)
        for issue, percentage in issue_percentages.items()
    }

    # Log results
    with open(log_file, "a") as log:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n\n<<< Feasibility Analysis Results >>>\n")
        log.write(f"Timestamp: {current_time}\n")
        log.write(f"Total number of cases: {total_cases}\n")
        log.write(f"Number of unacceptable cases: {num_unacceptable_cases}\n")
        log.write(
            f"Overall percentage of unacceptable cases: {overall_percentage:.2f}%\n")
        log.write(
            f"95% CI for overall percentage: ({overall_percentage_ci[0]:.2f}%, {overall_percentage_ci[1]:.2f}%)\n")

        # Log detailed percentages and CIs
        log.write(f"\nDetailed Results:\n")
        for issue, percentage in issue_percentages.items():
            ci_lower, ci_upper = issue_percentages_ci[issue]
            log.write(
                f"Percentage of cases with unacceptable {issue.lower()}: {percentage:.2f}%\n")
            log.write(f"  95% CI: ({ci_lower:.2f}%, {ci_upper:.2f}%)\n")

        # Log individual cases for each issue
        for issue, cases in issue_counts.items():
            if not cases.empty:
                log.write(f"\nCases with unacceptable {issue.lower()}:\n")
                log.writelines(
                    [f" - {pid}\n" for pid in cases['PatientID'].dropna()]
                )

        # Determine and log feasibility status
        feasibility_status = "Feasible" if overall_percentage < 10 else "Not Feasible"
        log.write(f"\nFeasibility Status: '{feasibility_status}'\n")
        log.write(f"-------------------------------------------------\n")

    return feasibility_status
