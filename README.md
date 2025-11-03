# POT Study Analysis

## Overview
This project contains code to analyze data from the investigated AI-driven software proof-of-technology Study.
The proof-of-technology is a pilot, single-site, prospective clinical study aims to evaluate the feasibility, safety, and diagnostic performance of the investigated AI-driven software compared to expert radiologist interpretation using PI-RADS v2.1 and histopathological findings from systematic and targeted biopsies.

## Contents

- **`data/`**: Folder containing raw scans and exported clinical data.
- **`scripts/`**: Scripts used for data preparation, cleaning, analysis, and visualization.
- **`results/`**: Output files such as tables, graphs, and summary statistics.
- **`main.py`**: Main python script to run the entire analysis pipeline.
- **`requirements.txt`**: File with dependencies must be installed to run the analysis.
- **`README.md`**: Overview of the project and usage instructions (this file).

## Requirements

To run the analysis, the following dependencies must be installed:

- **Programming Language**: Python (version 3.11)
- **Libraries**:
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - statsmodels
  - matplotlib
  - seaborn
  - pydicom

You can install the required libraries using the command:
```bash
pip install -r requirements.txt
```

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/ntnu-mr-cancer/POT-Analysis.git
```

2. Navigate to the project directory:
```bash
cd POT-Analysis
```

3. Set up the Python environment and install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place scans folders in the **`data/scans`** folder.
2. Place exported clinical excel sheets from eForsk (eCRFs) in the **`data/sheets`** folder.
3. Chek and adjust, if needed, the sheets paths in **`main.py`**.

### Analysis
Run the primary analysis script to generate results:
```bash
python main.py
```

### Outputs
Analysis output data saved in **`results/`**. The subdolsers are:
- **`tables/`**: Contains processed data tables in excel sheets. 
- **`reports/`**: Contains log files summerize the results for each analysis task.
- **`plots/`**: Contains the generated plots.

## File Descriptions

- **`scripts/analysis_pipeline.py`**: Run the entire analysis pipeline and generates outputs.
- **`scripts/data_preparation.py`**: Prepares the raw data for analysis.
- **`scripts/acquisition_parameters.py`**: Genrate summary of bpMRI acquisition parameters.
- **`scripts/characteristics_statistics.py`**: Generates general characteristics descriptive statistics.
- **`scripts/findings_statistics.py`**: Generates radiologist and AI findings statistics.
- **`scripts/feasability_analysis.py`**: Analyses AI feaability.
- **`scripts/safety_analysis.py`**: Analyses AI safety.
- **`scripts/patient_level_performance.py`**: Analyse patient-level performance.
- **`scripts/lesion_performance.py`**: Analyse lesion-level performance.
- **`scripts/comparison.py`**: Performs additional statistical comparisions.
- **`scripts/utils.py`**: General utilities.
- **`data/scans/`**: Contains the input bpMRI scans folders.
- **`data/sheets/`**: Contains the input exported clinical excel sheets from eForsk (eCRFs).
- **`results/tables/`**: Contains output processed data tables in excel sheets. 
- **`results/reports/`**: Contains output log files summerize the results for each analysis task.
- **`results/plots/`**: Contains output the generated plots.
- **`main.py`**: Main python script to run the entire analysis pipeline.
- **`requirements.txt`**: File with dependencies must be installed to run the analysis.
- **`README.md`**: Overview of the project and usage instructions (this file).

## License

This project is licensed under a Custom Proprietary License. Use of this analysis code is allowed only for personal and educational purposes. Commercial use or distribution without prior written consent is strictly prohibited. See the LICENSE file for full details.

## Contact

For questions or support, please contact Mohammed R. S. Sunoqrot at mohammed.sunoqrot@ntnu.no.

