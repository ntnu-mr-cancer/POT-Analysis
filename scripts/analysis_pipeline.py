"""
Main Analysis Code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import logging
import pandas as pd
from pathlib import Path
from scripts.data_preparation import prepare_data, prepare_adjusted_data
from scripts.acquisition_parameters import extract_mri_acquisition_parameters
from scripts.characteristics_statistics import generate_characteristics_statistics
from scripts.findings_statistics import generate_findings_statistics
from scripts.feasability_analysis import analyze_feasibility
from scripts.safety_analysis import analyze_safety
from scripts.patient_level_performance import evaluate_patient_level_performance
from scripts.lesion_performance import evaluate_lesion_level_performance
from scripts.comparison import run_comparisons


def setup_logging():
    """
    Sets up basic logging for the analysis pipeline.
    Logging provides information about execution status and errors.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def save_dataframe_to_excel(dataframe, file_path):
    """
    Saves a Pandas DataFrame to an Excel file.

    Parameters:
        dataframe (pd.DataFrame): The data to save.
        file_path (Path): Path to save the Excel file.

    Returns:
        None
    """
    try:
        # Save the DataFrame to the specified Excel file
        dataframe.to_excel(file_path, index=False)
        logging.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        # Log an error if saving fails
        logging.error(f"Failed to save data to {file_path}: {e}")


def integrate_mri_metadata(prepared_data, tables_dir, reports_dir, metadata_file="mri_acquisition_metadata.xlsx", log_file="acquisition_parameters.log"):
    """
    Integrates MRI acquisition metadata into the prepared data.

    Parameters:
        prepared_data (pd.DataFrame): The dataset to update with MRI metadata.
        tables_dir (Path): Directory to look for and save the metadata file.
        reports_dir (Path): Directory to save reports.
        metadata_file (str, optional): Name of the metadata Excel file. Defaults to "mri_acquisition_metadata.xlsx".
        log_file (str, optional): Name of the log file for extraction. Defaults to "acquisition_parameters.log".

    Returns:
        prepared_data (pd.DataFrame): The updated prepared_data with MRI scan dates.
    """
    metadata_path = tables_dir / metadata_file
    if metadata_path.is_file():
        # If the metadata file already exists, load it
        metadata_table = pd.read_excel(metadata_path)
    else:
        # If the metadata file doesn't exist, extract MRI parameters and save them
        metadata_table = extract_mri_acquisition_parameters(
            prepared_data, reports_dir / log_file
        )
        save_dataframe_to_excel(metadata_table, metadata_path)

    # Format scan dates and integrate them into the prepared dataset
    metadata_table['T2W.Scan Date'] = pd.to_datetime(
        metadata_table['T2W.Scan Date'], format='%Y%m%d').dt.strftime('%d.%m.%Y')
    prepared_data['ScanDate'] = metadata_table['T2W.Scan Date'].values

    return prepared_data


def create_output_directories(output_dir):
    """
    Creates required output subdirectories: reports, tables, and plots.

    Parameters:
        output_dir (Path): The main output directory.

    Returns:
        dict: A dictionary of created subdirectories.
    """
    # List of subdirectories to create
    subdirs = ['reports', 'tables', 'plots']
    for subdir in subdirs:
        # Create each subdirectory under the output directory
        path = output_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
    return {subdir: output_dir / subdir for subdir in subdirs}


def run_analysis_pipeline(data_path, clinical_metadata_path, radiology_metadata_path, ai_score_path,
                          scans_dir, output_dir, exclude_patient_ids=None, exclude_patient_performance_ids=None,
                          exception_patient_ids=None, additional_technical_issues_ids=None):
    """
    Executes the full analysis pipeline for the Study.

    Parameters:
        data_path (str): Path to the main data Excel file exported from eCRFs.
        clinical_metadata_path (str): Path to the general clinical metadata Excel file.
        radiology_metadata_path (str): Path to the radiology metadata Excel file.
        ai_score_path (str): Path to the manual Excel file for AI scores.
        scans_dir (str): Path to the folder containing mpMRI scans.
        output_dir (str): Path to the folder where analysis results and output will be saved.
        exclude_patient_ids (list, optional): Patient IDs to exclude from the analysis.
        exclude_patient_performance_ids (list, optional): Patient IDs to exclude from performance analysis.
        exception_patient_ids (list, optional): Patient IDs to include regardless of filters.
        additional_technical_issues_ids (list, optional): Patient IDs with additional technical issues.

    Returns:
        None
    """
    exclude_patient_ids = exclude_patient_ids or []
    exception_patient_ids = exception_patient_ids or []

    # Convert output directory to a Path object and create subdirectories
    output_dir = Path(output_dir)
    paths = create_output_directories(output_dir)

    try:
        # Step 1: Prepare data
        logging.info("Step 1: Preparing original data...")
        prepared_data, mapped_cleaned_data_for_feasability_safety = prepare_data(
            data_path, clinical_metadata_path, radiology_metadata_path,
            ai_score_path, scans_dir, exclude_patient_ids, exclude_patient_performance_ids,
            exception_patient_ids, additional_technical_issues_ids
        )

        # Step 2: Integrate MRI metadata
        logging.info(
            "Step 2: Extracting and integrating MRI acquisition metadata...")
        # Extract, integrate, and save MRI acquisition metadata for the cases used in performance analysis
        prepared_data = integrate_mri_metadata(
            prepared_data, paths['tables'], paths['reports'], "mri_acquisition_metadata.xlsx", "acquisition_parameters.log")

        save_dataframe_to_excel(
            prepared_data, paths['tables'] / "original_prepared_data.xlsx")

        # Extract, integrate, and save MRI acquisition metadata for the cases used in performance analysis
        integrate_mri_metadata(
            prepared_data, paths['tables'], paths['reports'], "mri_acquisition_metadata_for_all_cases.xlsx", "acquisition_parameters_for_all_cases.log")

        save_dataframe_to_excel(
            mapped_cleaned_data_for_feasability_safety, paths['tables'] / "original_prepared_data_for_feasability_and_safety.xlsx")

        # Step 3: Generate characteristics statistics
        logging.info("Step 3: Generating characteristics statistics...")
        generate_characteristics_statistics(
            prepared_data, paths['reports'] / "characteristics_statistics.log")

        # Step 4: Generate findings statistics
        logging.info("Step 4: Generating findings statistics...")
        original_findings_statistics = generate_findings_statistics(
            prepared_data, paths['reports'] /
            "original_findings_statistics.log"
        )

        # Step 5: Analyze feasibility
        logging.info("Step 5: Analyzing feasibility...")
        analyze_feasibility(
            mapped_cleaned_data_for_feasability_safety,
            paths['reports'] / "feasibility_analysis.log")

        # Step 6: Analyze safety
        logging.info("Step 6: Analyzing safety...")
        analyze_safety(mapped_cleaned_data_for_feasability_safety,
                       paths['reports'] / "safety_analysis.log")

        # Step 7: Evaluate patient-level performance
        logging.info("Step 7: Evaluating patient-level performance...")
        roc_data, operating_points = evaluate_patient_level_performance(
            original_findings_statistics, paths['reports'] /
            "patient_performance.log", paths['plots']
        )

        # Step 8: Evaluate lesion-level performance
        logging.info("Step 8: Evaluating lesion-level performance...")
        lesions_df, _ = evaluate_lesion_level_performance(
            prepared_data, operating_points, paths['reports'] /
            "lesion_performance.log", paths['plots']
        )
        save_dataframe_to_excel(
            lesions_df, paths['tables'] / "lesion_data.xlsx")

        # Step 9: Generate adjusted statistics
        logging.info("Step 9: Generating adjusted descriptive statistics...")
        prepared_adjusted_data, prepared_adjusted_data_full = prepare_adjusted_data(
            prepared_data, operating_points['Optimized thd']
        )
        save_dataframe_to_excel(prepared_adjusted_data,
                                paths['tables'] / "adjusted_prepared_data.xlsx")
        save_dataframe_to_excel(prepared_adjusted_data_full,
                                paths['tables'] / "adjusted_prepared_data_full.xlsx")

        # Generate findings statistics for adjusted data
        adjusted_findings_statistics = generate_findings_statistics(
            prepared_adjusted_data, paths['reports'] / "adjusted_findings_statistics.log")

        # Step 10: Run statistical comparisons
        logging.info("Step 10: Running statistical comparisons...")
        run_comparisons(
            original_findings_statistics, adjusted_findings_statistics,
            paths['reports'] / "comparisons_results.log"
        )

        logging.info("<<<--- Analysis pipeline completed successfully --->>>")

    except Exception as e:
        # Log any errors encountered during execution
        logging.error(f"Error during pipeline execution: {e}")
