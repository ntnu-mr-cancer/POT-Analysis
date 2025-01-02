"""
Acquisition parameters code, part of the
Aalysis code for the POT Study

Author: Mohammed R. S. Sunoqrot
"""
import os
import re
import pandas as pd
import pydicom
from datetime import datetime
from collections import Counter


def extract_acquisition_metadata_from_series(series_path):
    """
    Extract acquisition metadata from a series of DICOM images.

    Parameters:
        series_path (str): Path to the directory containing DICOM files.

    Returns:
        metadata_dict (dict): Dictionary with metadata fields as keys and their corresponding values as strings.
    """
    metadata_dict = {
        'Scan Date': set(),
        'TR (ms)': set(),
        'TE (ms)': set(),
        'FOV (mm)': set(),
        'Pixel Spacing (mm)': set(),
        'Slice Thickness (mm)': set(),
        'Interslice Gap (mm)': set(),
        'Flip Angle (°)': set(),
        'B-values (s/mm2)': set(),
        'Sequence Name': set(),
        'Manufacturer': set(),
        'Field Strength (T)': set(),
        'Number of Averages': set()
    }

    if not os.path.isdir(series_path):
        print(f"Path is not a directory: {series_path}")
        return None

    # Iterate over all files in the series directory
    for root, dirs, files in os.walk(series_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.dcm', '.ima')):
                try:
                    # Load the DICOM file
                    dicom_data = pydicom.dcmread(file_path)

                    # Extract metadata with common DICOM tags
                    metadata_dict['Scan Date'].add(
                        dicom_data.get('AcquisitionDate', 'Not Available'))
                    metadata_dict['TR (ms)'].add(
                        dicom_data.get('RepetitionTime', 'Not Available'))
                    metadata_dict['TE (ms)'].add(
                        dicom_data.get('EchoTime', 'Not Available'))
                    # Calculate FOV as 'Columns x Rows' if possible
                    if 'Columns' in dicom_data and 'Rows' in dicom_data:
                        columns = dicom_data.Columns
                        rows = dicom_data.Rows
                        fov = f"{columns} x {rows}"
                    else:
                        fov = 'Not Available'
                    metadata_dict['FOV (mm)'].add(fov)
                    metadata_dict['Pixel Spacing (mm)'].add(
                        ', '.join(map(str, dicom_data.PixelSpacing)) if dicom_data.PixelSpacing else 'Not Available')
                    metadata_dict['Slice Thickness (mm)'].add(
                        dicom_data.get('SliceThickness', 'Not Available'))
                    metadata_dict['Interslice Gap (mm)'].add(
                        dicom_data.get('SpacingBetweenSlices', 'Not Available'))
                    metadata_dict['Flip Angle (°)'].add(
                        dicom_data.get('FlipAngle', 'Not Available'))
                    # Extract B-values
                    b_value = dicom_data.get('BValue', '')
                    if not b_value:
                        sequence_name = dicom_data.get('SequenceName', '')
                        # Extract B-values from SequenceName if it contains '_b'
                        match = re.findall(
                            r'_b(\d+)(?:_(\d+))?', sequence_name)
                        if match:
                            # Join the matches into a comma-separated string
                            # Flatten and join the matches into a comma-separated string
                            b_values = [
                                b for match in match for b in match if b]
                            b_value = ', '.join(b_values)
                    metadata_dict['B-values (s/mm2)'].add(
                        str(b_value) if b_value else 'Not Available')
                    metadata_dict['Sequence Name'].add(
                        dicom_data.get('SequenceName', 'Not Available'))
                    metadata_dict['Manufacturer'].add(
                        dicom_data.get('Manufacturer', 'Not Available'))
                    metadata_dict['Field Strength (T)'].add(
                        dicom_data.get('MagneticFieldStrength', 'Not Available'))
                    metadata_dict['Number of Averages'].add(
                        dicom_data.get('NumberOfAverages', 'Not Available'))
                except Exception as e:
                    print(
                        f"Error reading image metadata from {file_path}: {e}")

    # Convert sets to comma-separated strings
    for key in metadata_dict:
        values = list(metadata_dict[key])
        if len(values) > 1:
            # Join values with a comma if there are multiple unique values
            metadata_dict[key] = ', '.join(str(value) for value in values)
        elif values:
            # Single value case
            metadata_dict[key] = values[0]
        else:
            # No value case
            metadata_dict[key] = 'Not Available'

    return metadata_dict


def extract_mri_acquisition_parameters(prepared_data, log_file="acquisition_parameters.log"):
    """
    Extracts acquisition metadata (TR, TE, FOV, etc.) from MRI images for each patient in the prepared data.
    Summarizes and logs the analysis results.

    Parameters:
        prepared_data (pd.DataFrame): DataFrame containing the patient data and MRI paths.
        log_file (str): Path to save the analysis log file.

    Returns:
        metadata_table (pd.DataFrame): Table of extracted metadata for each patient.
    """
    # Initialize list to store extracted metadata
    extracted_metadata = []

    # Iterate over each patient in the prepared data
    for index, row in prepared_data.iterrows():
        patient_id = row['PatientID']
        print(f"Processing MRI metadata for PatientID: {patient_id}")

        # Extract image paths from the DataFrame
        t2w_path = row.get('T2WOriginalPath')
        adc_path = row.get('ADCOriginalPath')
        hbv_path = row.get('HBVOriginalPath')
        dwi_path = row.get('DWIOriginalPath')

        # Extract acquisition metadata for each modality
        t2w_metadata = extract_acquisition_metadata_from_series(
            t2w_path) if t2w_path else None
        adc_metadata = extract_acquisition_metadata_from_series(
            adc_path) if adc_path else None
        hbv_metadata = extract_acquisition_metadata_from_series(
            hbv_path) if hbv_path else None
        dwi_metadata = extract_acquisition_metadata_from_series(
            dwi_path) if dwi_path else None

        # Combine all metadata into a single record
        patient_metadata = {
            'PatientID': patient_id,
            'T2W': t2w_metadata,
            'ADC': adc_metadata,
            'HBV': hbv_metadata,
            'DWI': dwi_metadata
        }

        # Append the result to the list
        extracted_metadata.append(patient_metadata)

    # Convert the list to a DataFrame
    metadata_table = pd.json_normalize(extracted_metadata)

    # Log the summarized results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_cases = len(metadata_table)

    # Initialize logging
    with open(log_file, "a") as log:
        log.write(f"\n\n<<< MRI Acquisition Metadata Analysis >>>\n")
        log.write(f"Timestamp: {current_time}\n")
        log.write(f"Total number of cases analyzed: {total_cases}\n")

        # For each column, calculate and log the mean, median, min, max
        for column in metadata_table.columns:
            if column == 'PatientID':
                continue

            column_data = metadata_table[column].dropna()

            if not column_data.empty:
                # Convert to numeric for statistical operations
                column_data_numeric = pd.to_numeric(
                    column_data.apply(pd.Series).stack(), errors='coerce')

                if column_data_numeric.notna().any():
                    # Numeric Data
                    mean_value = round(column_data_numeric.mean(), 2)
                    std_value = round(column_data_numeric.std(), 2)
                    median_value = round(column_data_numeric.median(), 2)
                    min_value = round(column_data_numeric.min(), 2)
                    max_value = round(column_data_numeric.max(), 2)
                    q1 = round(column_data_numeric.quantile(0.25), 2)
                    q3 = round(column_data_numeric.quantile(0.75), 2)
                    iqr = round(q3 - q1, 2)
                    unique_values = column_data_numeric.value_counts()

                    log.write(f"\n{column} (Numeric):\n")
                    log.write(f"  Mean: {mean_value}\n")
                    log.write(f"  Std: {std_value}\n")
                    log.write(f"  Median: {median_value}\n")
                    log.write(f"  Min: {min_value}\n")
                    log.write(f"  Max: {max_value}\n")
                    log.write(f"  Q1: {q1}\n")
                    log.write(f"  Q3: {q3}\n")
                    log.write(f"  IQR: {iqr}\n")
                    log.write("  Unique values and frequencies:\n")
                    for value, count in unique_values.items():
                        log.write(f"    {value}: {count}\n")
                else:
                    # String Data
                    value_list = column_data.tolist()
                    counts = Counter(value_list)

                    unique_values = counts.items()
                    most_common_value = counts.most_common(
                        1)[0][0] if counts else 'Not Available'
                    least_common_value = counts.most_common(
                    )[-1][0] if counts else 'Not Available'

                    # Find median (most common if even number of items)
                    sorted_values = sorted(
                        value_list, key=lambda x: counts[x], reverse=True)
                    mid_index = len(sorted_values) // 2
                    median_value = sorted_values[mid_index] if len(
                        sorted_values) % 2 == 1 else sorted_values[mid_index - 1]

                    log.write(f"\n{column} (String):\n")
                    log.write(f"  Median: {median_value}\n")
                    log.write(f"  Most common: {most_common_value}\n")
                    log.write(f"  Least common: {least_common_value}\n")
                    log.write("  Unique values and frequencies:\n")
                    for value, count in unique_values:
                        log.write(f"    {value}: {count}\n")
            else:
                log.write(f"\n{column} contains no data.\n")

        log.write(f"\n-------------------------------------------------\n")

    return metadata_table
