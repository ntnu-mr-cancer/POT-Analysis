"""
Safety analysis code, part of the
Analysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import pandas as pd
from scripts.utils import calculate_confidence_interval
from datetime import datetime


def analyze_safety(prepared_data, log_file="safety_analysis.log"):
    """
    Analyzes safety based on reported events and event types, highlighting Serious Adverse Events (SAE),
    and logs detailed event information per patient.

    Parameters:
        prepared_data (pd.DataFrame): The prepared dataset to analyze.
        log_file (str): Path to the log file where results are written.

    Returns:
        safety_status (str): 'Safety Status: Safe (No SAE)' if no SAE reported, otherwise 'Safety Status: Not Safe (SAE reported)'.
    """
    # Record current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract rows where events were reported
    event_data = prepared_data[prepared_data["ExperiencedEvent"] == "Yes"]
    total_events = len(event_data)

    # Early exit for no events
    if total_events == 0:
        safety_status = "Safety Status: Safe (No events reported)"
        logs = [
            f"\n\n<<< Safety Analysis Results >>>",
            f"Timestamp: {current_time}",
            "No events reported.",
            safety_status,
            f"{'-' * 50}",
        ]
        with open(log_file, "a") as log:
            log.write("\n".join(logs) + "\n")
        return safety_status

    # Analyze event types
    event_type_columns = [
        col for col in prepared_data.columns if "EventType" in col]
    event_counts = {}
    for col in event_type_columns:
        counts = event_data[col].value_counts()
        for event_type, count in counts.items():
            event_counts[event_type] = event_counts.get(event_type, 0) + count

    # Calculate percentages for event types and check for SAE
    is_safe = True
    event_percentages = {
        event: (count / total_events) * 100 for event, count in event_counts.items()
    }
    if any("SAE" in event for event in event_counts):
        is_safe = False

    safety_status = "Safety Status: Safe (No SAE)" if is_safe else "Safety Status: Not Safe (SAE reported)"

    # Prepare detailed event logging per patient
    patient_event_details = []
    max_event_columns = max(
        int(col.split("EventType")[1]) for col in event_type_columns
    )  # Find max event index

    for _, row in event_data.iterrows():
        patient_id = row["PatientID"]
        for i in range(1, max_event_columns + 1):
            event_type_col = f"EventType{i}"
            if event_type_col in prepared_data.columns and pd.notna(row[event_type_col]):
                event_details = {
                    "PatientID": patient_id,
                    "EventType": row[event_type_col],
                    "Description": row.get(f"DescribeEvent{i}", "N/A"),
                    "StartDate": row.get(f"EventStartDate{i}", "N/A"),
                    "EndDate": row.get(f"EventEndDate{i}", "N/A"),
                    "Severity": row.get(f"EventSeverity{i}", "N/A"),
                    "Outcome": row.get(f"EventOutcome{i}", "N/A")
                }
                patient_event_details.append(event_details)

    # Add percentage and CI calculations
    total_patients = len(prepared_data)
    event_percentage = (total_events / total_patients *
                        100) if total_patients > 0 else 0
    event_percentage_ci = calculate_confidence_interval(
        event_percentage, total_patients)

    # Calculate SAE-specific stats
    sae_count = sum(event_counts.get(event, 0)
                    for event in event_counts if "SAE" in event)
    sae_percentage = (sae_count / total_patients *
                      100) if total_patients > 0 else 0
    sae_percentage_ci = calculate_confidence_interval(
        sae_percentage, total_patients)

    # Prepare logs
    logs = [
        f"\n\n<<< Safety Analysis Results >>>",
        f"Timestamp: {current_time}",
        f"Total events: {total_events}",
        *[f"{event}: {percentage:.2f}%" for event,
            percentage in event_percentages.items()],
        "\nEvent Details per Patient:",
        f"| {'PatientID':<15} | {'EventType':<20} | {'Description':<40} | "
        f"{'StartDate':<12} | {'EndDate':<12} | {'Severity':<15} | {'Outcome':<15} |",
        f"{'-' * 137}",
        *[
            f"| {detail['PatientID']:<15} | {detail['EventType']:<20} | {detail['Description']:<40} | "
            f"{detail['StartDate']:<12} | {detail['EndDate']:<12} | {detail['Severity']:<15} | {detail['Outcome']:<15} |"
            for detail in patient_event_details
        ],
        f"\n{safety_status}",
        f"Percentage of all events to total patients: {event_percentage:.2f}% "
        f"(95% CI: {event_percentage_ci[0]:.2f}%, {event_percentage_ci[1]:.2f}%)",
        f"Percentage of SAE to total patients: {sae_percentage:.2f}% "
        f"(95% CI: {sae_percentage_ci[0]:.2f}%, {sae_percentage_ci[1]:.2f}%)"
    ]

    # Write logs at the end
    with open(log_file, "a") as log:
        log.write("\n".join(logs) + "\n")

    return safety_status
