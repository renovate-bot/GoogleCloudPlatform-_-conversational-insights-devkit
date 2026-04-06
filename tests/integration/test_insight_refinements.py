# Copyright 2026 Google. This software is provided as-is, without warranty or representation for any use or purpose. Your use of it is subject to your agreement with Google.
"""
Integration Tests for QAI Pipeline Refactoring.
Tests the Adapter Pattern and integration with Devkit wrappers.
"""

import pytest
import unittest.mock as mock
from src.workflow.insight_refinements.schemas.input import (
    BatchAnalysisInput,
    GCPConfig,
    CCAIConfig,
    LLMConfig,
    AnalysisConfig,
    BigQueryConfig,
)
from src.workflow.insight_refinements.analysis import run_batch_analysis
from src.wrapper.insights import Analysis, Annotators


@pytest.fixture
def mock_config():
    """Provides a standard configuration for testing."""
    return BatchAnalysisInput(
        gcp=GCPConfig(
            project_id="insights-python-tooling-prober",
            project_number="123456789",
            location_id="us-central1",
        ),
        ccai=CCAIConfig(
            scorecard_id="test-scorecard-id",
        ),
        llm=LLMConfig(
            model_name="gemini-3.1-flash-lite-preview",
            location_id="us-central1",
        ),
        analysis=AnalysisConfig(
            analysis_percentage=10,
            enable_bulk_analysis=True,
            filter="labels.test=true",
            annotator_selector={
                "run_intent_annotator": True,
                "run_entity_annotator": True,
                "run_summarization_annotator": True,
            }
        ),
        bigquery=BigQueryConfig(
            dataset_id="test_dataset",
            staging_table_id="staging_table",
            main_table_id="main_table",
        ),
    )


def test_run_batch_analysis_adapter_wiring(mock_config):
    """
    Verifies that run_batch_analysis correctly translates the 
    Pydantic config into Devkit wrapper calls.
    """
    with mock.patch("src.workflow.insight_refinements.analysis.Analysis") as MockAnalysis, \
         mock.patch("src.workflow.insight_refinements.analysis.qai_logger") as mock_logger:
        # Mock the bulk method to return a dummy operation
        mock_instance = MockAnalysis.return_value
        mock_op = mock.MagicMock()
        mock_op.name = "mock-operation-name"
        mock_op.result.return_value = {"status": "mock-completed"}
        mock_instance.bulk.return_value = mock_op

        # Execute the refactored function
        results = run_batch_analysis(mock_config)

        # Assert results
        assert results[0]["status"] == "success"
        
        # Verify Analysis was instantiated with the correct adapter objects
        MockAnalysis.assert_called_once()
        _, kwargs = MockAnalysis.call_args
        
        # Check that auth and config were passed (Adapter Pattern validation)
        assert "auth" in kwargs
        assert "config" in kwargs
        assert kwargs["config"].region == "us-central1"
        
        # Verify bulk was called with translated Enums
        mock_instance.bulk.assert_called_once()
        bulk_args, bulk_kwargs = mock_instance.bulk.call_args
        
        # Check annotators translation: intent/entity -> INSIGHTS, summarization -> SUMMARIZATION
        annotators = bulk_kwargs["annotators"]
        assert Annotators.INSIGHTS in annotators
        assert Annotators.SUMMARIZATION in annotators
        assert Annotators.TOPIC_MODEL not in annotators
        
        # Check other bulk params
        assert bulk_kwargs["analysis_percentage"] == 10.0
        assert bulk_kwargs["analysis_filter"] == "labels.test=true"


def test_annotator_translation_edge_cases(mock_config):
    """Verifies different combinations of annotator selectors map correctly."""
    from src.workflow.insight_refinements.analysis import _annotator_dict_to_enum_list

    # 1. Just Topic Model
    sel1 = {"run_issue_model_annotator": True}
    res1 = _annotator_dict_to_enum_list(sel1)
    assert res1 == [Annotators.TOPIC_MODEL]

    # 2. Mixed Insights
    sel2 = {"run_intent_annotator": True, "run_sentiment_annotator": True}
    res2 = _annotator_dict_to_enum_list(sel2)
    assert res2 == [Annotators.INSIGHTS]

    # 3. Empty
    res3 = _annotator_dict_to_enum_list({})
    assert res3 == []
