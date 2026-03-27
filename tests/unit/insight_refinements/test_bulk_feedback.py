# Copyright 2026 Google. This software is provided as-is, without warranty or representation for any use or purpose. Your use of it is subject to your agreement with Google.
import sys
from unittest.mock import MagicMock, patch

# MOCK THE MISSING DEPENDENCY BEFORE IMPORT
mock_ccai = MagicMock()
sys.modules["google.cloud.contact_center_insights_v1"] = mock_ccai
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud"].contact_center_insights_v1 = mock_ccai

import pytest  # noqa: E402
from unittest.mock import mock_open  # noqa: E402
from src.workflow.insight_refinements.bulk_feedback import BulkFeedbackManager  # noqa: E402
from src.workflow.insight_refinements.schemas.input import BatchAnalysisInput  # noqa: E402


@pytest.fixture
def mock_config():
    return BatchAnalysisInput(
        **{
            "gcp": {
                "project_id": "test-p",
                "project_number": "12345",
                "location_id": "us-central1",
            },
            "ccai": {"scorecard_id": "123"},
            "llm": {"model_name": "gemini", "location_id": "us"},
            "bigquery": {
                "dataset_id": "d",
                "staging_table_id": "s",
                "main_table_id": "m",
            },
        }
    )


@patch("src.workflow.insight_refinements.utils.get_storage_client")
def test_csv_row_to_json_label(mock_storage, mock_config):
    """Test the mapping from CSV row to FeedbackLabel JSON structure."""
    # We don't need to patch CCAI client again because we mocked the module
    manager = BulkFeedbackManager(mock_config)

    # Test Case 1: Yes/No + Rationale
    row = {
        "conversation_id": "123",
        "question_id": "q1",
        "answer_value": "Yes",
        "rationale": "Good greeting",
        "score": "1.0",
    }

    label = manager._csv_row_to_json_label(row)

    assert (
        "projects/test-p/locations/us-central1/conversations/123"
        in label["labeled_resource"]
    )
    assert label["qa_answer_label"]["key"] == "q1"
    assert label["qa_answer_label"]["bool_value"] is True
    assert (
        label["qa_answer_label"]["qa_answer_rationale"]["rationale"] == "Good greeting"
    )
    assert label["qa_answer_label"]["score"] == 1.0


@patch("src.workflow.insight_refinements.utils.get_storage_client")
def test_upload_from_local_csv(mock_storage, mock_config):
    """Test the orchestration of reading CSV and uploading to GCS."""
    manager = BulkFeedbackManager(mock_config)

    # Mock CSV content
    csv_content = """conversation_id,question_id,answer_value,rationale,score
123,q1,Yes,reason,1.0"""

    # Mock Storage
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Mock Bulk Upload API call on the manager's client (which is a mock from sys.modules)
    mock_operation = MagicMock()
    mock_operation.result.return_value.upload_stats.success_count = 1
    manager.client.bulk_upload_feedback_labels.return_value = mock_operation

    with patch("builtins.open", mock_open(read_data=csv_content)):
        result = manager.upload_from_local_csv("dummy.csv", "my-bucket")

    assert result["status"] == "success"
    mock_blob.upload_from_string.assert_called_once()

    # Verify content
    uploaded_jsonl = mock_blob.upload_from_string.call_args[0][0]
    assert "qa_answer_rationale" in uploaded_jsonl
