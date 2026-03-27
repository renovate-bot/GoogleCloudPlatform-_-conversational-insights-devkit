# Copyright 2026 Google. This software is provided as-is, without warranty or representation for any use or purpose. Your use of it is subject to your agreement with Google.
import pytest
from src.workflow.qai_pipeline.schemas.input import BatchAnalysisInput, FeedbackCSVRow
from src.workflow.qai_pipeline.schemas.output import EvaluationMetrics, EvaluationResult
from src.workflow.qai_pipeline.schemas.topic import L1TopicDefinition, L2TaxonomyResult, TopicAuditLog


def test_batch_analysis_input_validation():
    """Test validation logic in BatchAnalysisInput."""
    # Valid config
    config_data = {
        "gcp": {
            "project_id": "p",
            "project_number": "123",
            "location_id": "l",
        },
        "ccai": {"scorecard_id": "123"},
        "llm": {"model_name": "gemini", "location_id": "us"},
        "bigquery": {"dataset_id": "d", "staging_table_id": "s", "main_table_id": "m"},
    }
    input_model = BatchAnalysisInput(**config_data)
    assert input_model.gcp.project_id == "p"

    # Invalid endpoint
    config_data["ccai"]["insights_endpoint"] = "invalid.com"
    with pytest.raises(ValueError):
        BatchAnalysisInput(**config_data)


def test_evaluation_metrics_validation():
    """Test validation logic in EvaluationMetrics."""
    metrics = EvaluationMetrics(
        accuracy=0.9,
        precision=0.8,
        recall=0.7,
        f1_score=0.75,
        cohens_kappa=0.6,
        total_samples=100,
    )
    assert metrics.accuracy == 0.9


def test_feedback_csv_row_validation():
    """Test validation for FeedbackCSVRow."""
    # Valid row
    row = FeedbackCSVRow(
        conversation_id="conv1",
        question_id="q1",
        answer_value="Yes",
        rationale="Agent greeted clearly.",
        score=1.0,
    )
    assert row.conversation_id == "conv1"
    assert row.rationale == "Agent greeted clearly."

    # Missing required field
    with pytest.raises(ValueError):
        FeedbackCSVRow(conversation_id="conv1")  # Missing question_id, answer_value


def test_evaluation_result_rationale():
    """Test that EvaluationResult accepts rationale."""
    result = EvaluationResult(
        resource_id="r1",
        question_id="q1",
        predicted_answer="Yes",
        golden_answer="No",
        is_correct=False,
        rationale="LLM missed the nuance.",
    )
    assert result.rationale == "LLM missed the nuance."


def test_l1_topic_definition_validation():
    """Test validation for L1TopicDefinition."""
    topic = L1TopicDefinition(
        issue_model_id="model-123",
        issue_id="issue-123",
        display_name="Greeting",
        description="Agent greets the customer.",
        exported_at="2026-03-18T12:00:00Z",
    )
    assert topic.issue_model_id == "model-123"
    assert topic.issue_id == "issue-123"


def test_l2_taxonomy_result_validation():
    """Test validation for L2TaxonomyResult."""
    result = L2TaxonomyResult(
        conversation_id="conv-1",
        l1_issue_id="issue-123",
        l2_category="Polite Greeting",
        reasoning="The agent used a very polite tone.",
    )
    assert result.l2_category == "Polite Greeting"


def test_topic_audit_log_validation():
    """Test validation for TopicAuditLog."""
    log = TopicAuditLog(
        run_id="run-789",
        conversation_id="conv-1",
        status="success",
        latency_ms=150.5,
        input_tokens=100,
        output_tokens=50,
    )
    assert log.status == "success"
    assert log.latency_ms == 150.5
