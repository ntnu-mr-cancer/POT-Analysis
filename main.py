"""
Main Analysis Code for the POT Study
Author: Mohammed R. S. Sunoqrot
"""
from pathlib import Path
import logging
from scripts.analysis_pipeline import run_analysis_pipeline


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def validate_paths(*paths):
    """Validate the existence of given file paths."""
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")


def get_paths():
    """Define and return all necessary paths."""
    parent_dir = Path(__file__).resolve().parent
    paths = {
        "data_path": parent_dir / "data" / "sheets" / "exported_eCRF_data.xlsx",
        "clinical_metadata_path": parent_dir / "data" / "sheets" / "clinical_eCRF_metadata.xlsx",
        "radiology_metadata_path": parent_dir / "data" / "sheets" / "radiology_eCRF_metadata.xlsx",
        "ai_score_path": parent_dir / "data" / "sheets" / "manual_ai_scores.xlsx",
        "scans_dir": parent_dir / "data" / "scans",
        "output_dir": parent_dir / "results"
    }
    return paths


if __name__ == "__main__":
    setup_logging()
    logging.info(">> Starting POT Study analysis pipeline...")

    try:
        # Load paths
        paths = get_paths()

        # Validate input paths
        validate_paths(
            paths["data_path"], paths["clinical_metadata_path"],
            paths["radiology_metadata_path"], paths["ai_score_path"],
            paths["scans_dir"]
        )

        # Patient exclusions
        exclude_patient_ids = ['POT0078', 'POT0050']
        exception_patient_ids = []

        # Run the analysis pipeline
        run_analysis_pipeline(
            data_path=str(paths["data_path"]),
            clinical_metadata_path=str(paths["clinical_metadata_path"]),
            radiology_metadata_path=str(paths["radiology_metadata_path"]),
            ai_score_path=str(paths["ai_score_path"]),
            scans_dir=str(paths["scans_dir"]),
            output_dir=str(paths["output_dir"]),
            exclude_patient_ids=exclude_patient_ids,
            exception_patient_ids=exception_patient_ids
        )
        logging.info("<< Analysis pipeline completed successfully. >>")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
