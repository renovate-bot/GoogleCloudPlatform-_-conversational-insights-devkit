# Copyright 2026 Google. This software is provided as-is, without warranty or representation for any use or purpose. Your use of it is subject to your agreement with Google.
import pytest
from src.workflow.qai_pipeline.schemas.input import BatchAnalysisInput


def test_config_with_new_fields():
    """Test that the updated config schema correctly parses new fields."""
    config_data = {
        "gcp": {
            "project_id": "test-project",
            "project_number": "123456789",
            "location_id": "us-central1",
            "staging_bucket": "my-bucket",
        },
        "ccai": {
            "scorecard_id": "scorecard-123",
            "feedback_regionalization": True,
        },
        "llm": {
            "model_name": "gemini-3.0-flash",
            "location_id": "us-central1",
        },
        "bigquery": {
            "project_id": "analytics-project",
            "dataset_id": "test_dataset",
            "staging_table_id": "staging",
            "main_table_id": "main",
        },
        "topic_refinement": {
            "issue_model_id": "model-abc",
            "prompt_gcs_uri": "gs://bucket/prompt.txt",
        },
    }

    config = BatchAnalysisInput(**config_data)

    assert config.gcp.project_number == "123456789"
    assert config.gcp.staging_bucket == "my-bucket"
    assert config.ccai.feedback_regionalization is True
    assert config.bigquery.project_id == "analytics-project"
    assert config.topic_refinement.issue_model_id == "model-abc"
    assert config.topic_refinement.prompt_gcs_uri == "gs://bucket/prompt.txt"
    # Verify defaults
    assert config.topic_refinement.l1_definitions_table == "l1_topic_definitions"


def test_config_missing_required_fields():
    """Test that missing required fields raises validation error."""
    invalid_data = {
        "gcp": {"project_id": "p", "location_id": "l"},  # Missing project_number
        "ccai": {"scorecard_id": "s"},
        "llm": {"model_name": "m", "location_id": "l"},
        "bigquery": {"dataset_id": "d", "staging_table_id": "s", "main_table_id": "m"},
    }

    with pytest.raises(ValueError):
        BatchAnalysisInput(**invalid_data)
