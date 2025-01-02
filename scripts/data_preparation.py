"""
Data preparation code, part of the
Analysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import os
import re
import pandas as pd
from pathlib import Path


def get_latest_folder(patient_folder):
    """
    Get the subfolder with the latest creation date from the given patient folder.

    Parameters:
        patient_folder (str): Path to the folder containing patient subfolders.

    Returns:
        os.DirEntry: Subfolder with the latest creation date, or None if no subfolders exist.
    """
    # List all subfolders within the patient folder
    subfolders = [f for f in os.scandir(patient_folder) if f.is_dir()]
    # Return the folder with the most recent creation time
    return max(subfolders, key=lambda f: f.stat().st_ctime, default=None)


def find_series_folder(base_folder, series_keywords):
    """
    Find the folder that contains the desired series based on the keywords provided.
    If there are multiple folders, select the one with the highest number extracted from
    the last two characters of the folder name. Exclude folders with names containing
    'test', 'raskere', or 'ny' (case-insensitive).

    Parameters:
        base_folder (str): Path to the base folder to search.
        series_keywords (list): List of keywords to identify the desired series folder.

    Returns:
        os.DirEntry: The folder matching the criteria, or None if no match is found.
    """
    # Keywords to exclude from folder names
    exclude_keywords = {'test', 'raskere', 'ny'}

    def folder_matches(folder_name):
        """Check if the folder name matches the keywords and excludes certain keywords."""
        name_lower = folder_name.lower()
        return (
            all(kw.lower() in name_lower for kw in series_keywords) and
            not any(ex_kw in name_lower for ex_kw in exclude_keywords)
        )

    # Identify matching folders
    matching_folders = [
        f for f in os.scandir(base_folder) if f.is_dir() and folder_matches(f.name)
    ]

    if not matching_folders:
        return None

    def extract_number(folder_name):
        """Extract the last two characters of the folder name as a number."""
        match = re.search(r'(\d{2})$', folder_name)
        return int(match.group(1)) if match else 0

    # Return folder with the highest extracted number
    return max(matching_folders, key=lambda f: extract_number(f.name))


def find_mri_series_paths(patient_id, scans_dir):
    """
    Find the T2W, ADC, HBV, and DWI series paths for the given patient.
    The function checks the main folder, "WithdrawConsent", and "Excluded" folders.

    Parameters:
        patient_id (str): The ID of the patient to search for.
        scans_dir (str): Path to the master folder containing MRI scans.

    Returns:
        dict: A dictionary containing the paths to the T2W, ADC, HBV, and DWI series, or None if not found.
    """
    series_paths = {
        'T2WOriginalPath': None,
        'ADCOriginalPath': None,
        'HBVOriginalPath': None,
        'DWIOriginalPath': None
    }

    # Subfolders to search
    subfolders_to_check = ['', 'WithdrawConsent', 'Excluded']

    for subfolder in subfolders_to_check:
        search_path = Path(scans_dir) / subfolder
        if not search_path.exists():
            continue

        # Find folders containing the PatientID in their name
        patient_folders = [
            f for f in os.scandir(search_path) if f.is_dir() and patient_id in f.name
        ]

        if not patient_folders:
            continue

        # Select the most recent patient folder
        patient_folder = max(patient_folders, key=lambda f: f.stat().st_ctime)

        # Get the latest subfolder inside the patient's folder
        latest_folder = get_latest_folder(patient_folder.path)
        if not latest_folder:
            continue

        # Locate specific series in the latest subfolder
        series_paths['T2WOriginalPath'] = find_series_folder(
            latest_folder.path, ['T2W_TRA']
        )
        series_paths['ADCOriginalPath'] = find_series_folder(
            latest_folder.path, ['ADC']
        )
        series_paths['HBVOriginalPath'] = find_series_folder(
            latest_folder.path, ['BVal']
        )
        series_paths['DWIOriginalPath'] = find_series_folder(
            latest_folder.path, ['DWI', 'TRACEW']
        )

        break  # Stop searching once the desired folder is found

    # Convert Path objects to strings
    return {key: str(path.path) if path else None for key, path in series_paths.items()}


def map_and_clean_data(data_path, general_metadata_path, radiology_metadata_path, ai_score_path, scans_dir, exclude_patient_ids=[], exception_patient_ids=[]):
    """
    Prepare the data to be able to use later.
    Maps values in the main data to their corresponding descriptions using general
    and radiology metadata. Cleans the data by removing unnecessary columns, filters by `HasCompleted`,
    excludes rows based on `PatientId`, and sorts the final data by `PatientId`.

    Parameters:
        data_path (str): Path to the main data Excel file.
        general_metadata_path (str): Path to the metadata Excel file for clinical variables.
        radiology_metadata_path (str): Path to the metadata Excel file for radiology evaluation.
        ai_score_path (str): Path to the manual AI scores Excel file.
        scans_dir (str): Path to the master folder containing mpMRI scans.
        exclude_patient_ids (list): List of Patient IDs to exclude from the final data.
        exception_patient_ids (list): Patient IDs to retain regardless of `HasCompletedStudy` status.

    Returns:
        mapped_cleaned_data (pd.DataFrame): A cleaned, mapped, filtered, and sorted DataFrame.
    """

    # Read the main data and metadata from Excel files into DataFrames
    data_df = pd.read_excel(data_path)
    general_metadata_df = pd.read_excel(general_metadata_path)
    radiology_metadata_df = pd.read_excel(radiology_metadata_path)
    manual_df = pd.read_excel(ai_score_path)

    # Concatenate the metadata DataFrames
    metadata_df = pd.concat(
        [general_metadata_df, radiology_metadata_df], ignore_index=True)

    # Create a dictionary to store mappings for each column
    column_mappings = {}

    # Build the mapping dictionary from metadata
    current_variable = None
    for _, row in metadata_df.iterrows():
        col = row['Variabelnavn']
        mapping_id = row['Alternativ ID']
        mapping_text = row['Alternativtekst']

        if col not in column_mappings:
            column_mappings[col] = {}

        if pd.notna(row['Variabelnavn']):
            current_variable = row['Variabelnavn']

        if pd.notna(mapping_id):
            column_mappings[current_variable][mapping_id] = mapping_text

    # Define a function to apply mappings
    def apply_mapping(row):
        for col in row.index:
            if col in column_mappings:
                mapping_dict = column_mappings[col]
                if pd.notna(row[col]):
                    row[col] = mapping_dict.get(row[col], row[col])
        return row

    # Apply the mapping function
    mapped_data = data_df.apply(apply_mapping, axis=1)

    # Define a function to merge non-empty values in each column
    def combine_values(series):
        non_empty_values = series.dropna().unique()
        return ', '.join(map(str, non_empty_values)) if len(non_empty_values) > 0 else None

    # Group by 'ForskningsobjektUnikNokkel' and apply the combine function
    grouped_data = mapped_data.groupby(
        'ForskningsobjektUnikNokkel', as_index=False).agg(combine_values)
    grouped_data = grouped_data.copy()  # De-fragment DataFrame after aggregation

    # Filter the data where 'HasCompletedStudy' is 'Yes', but allow exceptions

    filtered_data = grouped_data[
        (grouped_data['HasCompletedStudy'] == 'Yes') |
        (grouped_data['PatientID'].isin(exception_patient_ids)) |
        ((grouped_data['HasCompletedStudy'] == 'No')
         & (grouped_data['HasLeftStudy'] == 'No'))
    ]

    # Exclude rows where 'PatientID' is in the exclude_patient_ids list
    mapped_cleaned_data = filtered_data[~filtered_data['PatientID'].isin(
        exclude_patient_ids)]

    # Move 'PatientID' to the first column
    cols = ['PatientID'] + \
        [col for col in mapped_cleaned_data.columns if col != 'PatientID']
    mapped_cleaned_data = mapped_cleaned_data[cols]

    # Preserve the original column order before adding AI scores
    original_column_order = mapped_cleaned_data.columns.tolist()

    # Add AIScore columns if "AI" is found in any FindingSource column
    def add_ai_scores(row):
        finding_sources = ['FindingSource1',
                           'FindingSource2', 'FindingSource3',
                           'FindingSource4', 'FindingSource5',
                           'FindingSource6', 'FindingSource7']
        for i, source in enumerate(finding_sources, 1):
            if pd.notna(row[source]) and "AI" in row[source]:
                finding_nr = i
                score = manual_df.query(
                    f'patID == "{row["PatientID"]}" and findingNr == {finding_nr}')['ProvizScore']
                if not score.empty:
                    row[f'AIScore{finding_nr}'] = score.values[0]
                else:
                    row[f'AIScore{finding_nr}'] = None
        return row

    # Apply the function to add AI scores
    mapped_cleaned_data = mapped_cleaned_data.apply(add_ai_scores, axis=1)

    # Get the list of new AI score columns
    ai_score_columns = [
        col for col in mapped_cleaned_data.columns if col.startswith('AIScore')]

    # Reorder the DataFrame: 'PatientID' first, and AI score columns at the end
    final_columns = ['PatientID'] + \
        [col for col in original_column_order if col !=
            'PatientID'] + ai_score_columns
    mapped_cleaned_data = mapped_cleaned_data[final_columns]

    # Sort the DataFrame by 'PatientID' in ascending order
    mapped_cleaned_data = mapped_cleaned_data.sort_values(
        by='PatientID', ascending=True)

    # Add columns for the MRI paths
    # Collect all new column data in a list of dictionaries
    new_data = []

    for idx, row in mapped_cleaned_data.iterrows():
        patient_id = row['PatientID']
        series_paths = find_mri_series_paths(
            str(patient_id), scans_dir)
        new_data.append(series_paths)

    # Convert the list of dictionaries to a DataFrame
    new_columns_df = pd.DataFrame(new_data)

    # Concatenate the new columns with the original DataFrame
    mapped_cleaned_data = pd.concat(
        [mapped_cleaned_data.reset_index(drop=True), new_columns_df], axis=1)

    return mapped_cleaned_data


def correct_data(data):
    """
    Perform additional corrections for the prepared data, such as removing columns
    and updating specific patient information.

    Parameters:
        data (pd.DataFrame): The input DataFrame to be corrected.

    Returns:
       corrected_data (pd.DataFrame): The corrected DataFrame.
    """
    def apply_corrections(patient_id, columns_to_delete, updates):
        """Apply corrections for a specific patient."""
        if columns_to_delete:
            data.loc[data['PatientID']
                     == patient_id, columns_to_delete] = None
        for col, value in updates.items():
            data.loc[data['PatientID']
                     == patient_id, col] = value

    corrections = [
        # 1. Deal with radiology findings declared rejected but their values entered
        # POT0018
        {"patient_id": "POT0018", "columns_to_delete": [
            "FindingSource4", "FindingSide4", "FindingRegion4", "FindingLocation4", "AIScore4"], "updates": {"FindingsNumber": 3}},
        # 2. Deal with findings with thrshould less than 0.73 (used after correction in study)
        # POT0001 > Finding 1
        {"patient_id": "POT0001", "columns_to_delete": ["FindingSource1", "FindingSide1", "FindingRegion1", "FindingLocation1", "AIScore1",
                                                        "LesionOverallGGGTargeted1", "NumberofCoresTargeted1", "TargetedBiopsySide1",
                                                        "FindingAdditionalLocation1", "AdditionalSide1", "AdditionalRegion1", "AdditionalLocation1",
                                                        "TargetedBiopsyRegion1", "TargetedBiopsyLocation1", "TargetedBiopsyAdditionalLocation1",
                                                        "AdditionalTargetedBiopsySide1", "AdditionalTargetedBiopsyRegion1", "AdditionalTargetedBiopsyLocation1"],
         "updates": {"NumberofTargetedBiopsies": 0, "FindingsNumber": 0}},
        # POT0007 > Findings 1,2,and 3
        {"patient_id": "POT0007", "columns_to_delete": ["FindingSource1", "FindingSide1", "FindingRegion1", "FindingLocation1", "AIScore1",
                                                        "LesionOverallGGGTargeted1", "NumberofCoresTargeted1", "TargetedBiopsySide1",
                                                        "FindingSource2", "FindingSide2", "FindingRegion2", "FindingLocation2", "AIScore2",
                                                        "LesionOverallGGGTargeted2", "NumberofCoresTargeted2", "TargetedBiopsySide2",
                                                        "FindingSource3", "FindingSide3", "FindingRegion3", "FindingLocation3", "AIScore3",
                                                        "LesionOverallGGGTargeted3", "NumberofCoresTargeted3", "TargetedBiopsySide3"],
         "updates": {"NumberofTargetedBiopsies": 0, "FindingsNumber": 0}},
        # POT0010 > Finding 3
        {"patient_id": "POT0010", "columns_to_delete": ["FindingSource3", "FindingSide3", "FindingRegion3", "FindingLocation3", "AIScore3",
                                                        "LesionOverallGGGTargeted3", "NumberofCoresTargeted3", "TargetedBiopsySide3",
                                                        "TargetedBiopsyRegion3", "TargetedBiopsyLocation3", "TargetedBiopsyAdditionalLocation3"],
         "updates": {"NumberofTargetedBiopsies": 2, "FindingsNumber": 2}},
        # POT0012 > Finding 2
        {"patient_id": "POT0012", "columns_to_delete": ["FindingSource2", "FindingSide2", "FindingRegion2", "FindingLocation2", "AIScore2",
                                                        "LesionOverallGGGTargeted2", "NumberofCoresTargeted2", "TargetedBiopsySide2",
                                                        "TargetedBiopsyRegion2", "TargetedBiopsyLocation2", "TargetedBiopsyAdditionalLocation3"],
         "updates": {"NumberofTargetedBiopsies": 1, "FindingsNumber": 1}},
        # 3. Deal with when targeted biopsies more than radiological findings
        # POT0030 > Will delete targeted biopsy 3 and 4, but rest targeted biopsy 2 to have hoghest GG of the 3
        # and > AIScore2 correct incorrectly manually recorded AI Scores
        {"patient_id": "POT0030", "columns_to_delete": ["LesionOverallGGGTargeted3", "NumberofCoresTargeted3", "TargetedBiopsySide3",
                                                        "TargetedBiopsyRegion3", "TargetedBiopsyLocation3", "TargetedBiopsyAdditionalLocation3",
                                                        "LesionOverallGGGTargeted4", "NumberofCoresTargeted4", "TargetedBiopsySide4",
                                                        "TargetedBiopsyRegion4", "TargetedBiopsyLocation4", "TargetedBiopsyAdditionalLocation4"],
         "updates": {"NumberofTargetedBiopsies": 2, "LesionOverallGGGTargeted2": 1, "AIScore2": 0.7559}},
        # 4. Deal with when targeted biopsies more than radiological findings
        # POT0056 > delete finding 2 (after checking)
        {"patient_id": "POT0056", "columns_to_delete": ["FindingSource2", "FindingSide2", "FindingRegion2", "FindingLocation2", "AIScore2",
                                                        "FindingAdditionalLocation2", "AdditionalSide2", "AdditionalRegion2", "AdditionalLocation2"],
         "updates": {"FindingsNumber": 1}},
        # 5. Deal with incorrectly manually recorded AI Scores
        # POT0076 > AIScore3
        {"patient_id": "POT0076", "columns_to_delete": [],
            "updates": {"AIScore3": 0.7574}}
    ]

    for correction in corrections:
        apply_corrections(
            correction["patient_id"], correction["columns_to_delete"], correction["updates"]
        )

    corrected_data = data
    return corrected_data


def prepare_data(data_path, general_metadata_path, radiology_metadata_path, ai_score_path, scans_dir, exclude_patient_ids=[], exception_patient_ids=[]):
    """
    Wrapper function to prepare and correct the data and save it to an Excel file.

    Parameters:
        data_path (str): Path to the main data Excel file.
        general_metadata_path (str): Path to the metadata Excel file for clinical variables.
        radiology_metadata_path (str): Path to the metadata Excel file for radiology evaluation.
        ai_score_path (str): Path to the manual AI scores Excel file.
        scans_dir (str): Path to the master folder containing mpMRI scans.
        exclude_patient_ids (list): List of Patient IDs to exclude from the final data.
        - exception_patient_ids (list): Patient IDs to retain regardless of `HasCompletedStudy` status.

    Returns:
        prepared_data (pd.DataFrame): prepared, corrected data.
    """
    # Prepare data by mapping and cleaning
    mapped_cleaned_data = map_and_clean_data(
        data_path, general_metadata_path, radiology_metadata_path, ai_score_path, scans_dir,
        exclude_patient_ids, exception_patient_ids
    )
    # Correct mistakes
    corrected_data = correct_data(mapped_cleaned_data)

    prepared_data = corrected_data

    return prepared_data


def prepare_adjusted_data(prepared_data, optimized_threshold):
    """
    Adjust the prepared data by evaluating specific columns and thresholds.

    Parameters:
        prepared_data (pd.DataFrame): The input DataFrame prepared by `prepare_data`.
        optimized_threshold (float): Threshold for AIScore values to adjust or remove rows.

    Returns:
        prepared_adjusted_data (pd.DataFrame): The adjusted DataFrame with adjustments applied.
        prepared_adjusted_data_full (pd.DataFrame): The adjusted DataFrame with adjustments applied full columns.
    """
    # Define the column patterns for processing
    for idx in range(1, 5):
        source_col = f"FindingSource{idx}"
        ai_score_col = f"AIScore{idx}"
        columns_to_delete = [
            f"FindingSource{idx}", f"FindingSide{idx}", f"FindingRegion{idx}", f"FindingLocation{idx}", f"AIScore{idx}",
            f"LesionOverallGGGTargeted{idx}", f"NumberofCoresTargeted{idx}", f"TargetedBiopsySide{idx}",
            f"FindingAdditionalLocation{idx}", f"AdditionalSide{idx}", f"AdditionalRegion{idx}", f"AdditionalLocation{idx}",
            f"TargetedBiopsyRegion{idx}", f"TargetedBiopsyLocation{idx}", f"TargetedBiopsyAdditionalLocation{idx}",
            f"AdditionalTargetedBiopsySide{idx}", f"AdditionalTargetedBiopsyRegion{idx}", f"AdditionalTargetedBiopsyLocation{idx}"
        ]

        for i, row in prepared_data.iterrows():
            if row[source_col] == "Human + AI Together" and row[ai_score_col] < optimized_threshold:
                # Rename "Human + AI Together" to "Human Alone" and remove AIScore value
                prepared_data.at[i, source_col] = "Human Alone"
                prepared_data.at[i, ai_score_col] = None
            elif row[source_col] == "AI Alone" and row[ai_score_col] < optimized_threshold:
                # Remove values and reduce counters
                prepared_data.at[i, source_col] = None
                for col in columns_to_delete:
                    prepared_data.at[i, col] = None
                if not pd.isnull(row["FindingsNumber"]):
                    prepared_data.at[i, "FindingsNumber"] -= 1
                if not pd.isnull(row["NumberofTargetedBiopsies"]):
                    prepared_data.at[i, "NumberofTargetedBiopsies"] -= 1

    prepared_adjusted_data_full = prepared_data
    # Drop the last 5 columns
    prepared_adjusted_data = prepared_data.iloc[:, :-5]

    return prepared_adjusted_data, prepared_adjusted_data_full
