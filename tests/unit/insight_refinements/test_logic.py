# Copyright 2026 Google. This software is provided as-is, without warranty or representation for any use or purpose. Your use of it is subject to your agreement with Google.
import pytest
import pandas as pd
from unittest.mock import patch
from src.workflow.insight_refinements.analysis import (
    ConversationAnalyzer,
    BigQueryOperator,
    run_batch_analysis,
)
from src.workflow.insight_refinements.evaluation import (
    QuestionEvaluation,
    ConversationEvaluator,
    run_automated_evaluation,
)
from src.workflow.insight_refinements.schemas.input import BatchAnalysisInput


@pytest.fixture
def mock_config():
    return BatchAnalysisInput(
        **{
            "gcp": {
                "project_id": "test-project",
                "project_number": "12345",
                "location_id": "us-central1",
            },
            "ccai": {"scorecard_id": "12345"},
            "llm": {"model_name": "gemini-1.5-pro", "location_id": "us-central1"},
            "bigquery": {
                "dataset_id": "test_dataset",
                "staging_table_id": "staging",
                "main_table_id": "main",
            },
        }
    )


def test_question_evaluation_correctness():
    """Test the is_correct property of QuestionEvaluation."""
    qe_true = QuestionEvaluation(
        question_id="1",
        question_text="Test?",
        golden_answer="Meets",
        predicted_answer="Meets",
    )
    assert qe_true.is_correct is True

    qe_false = QuestionEvaluation(
        question_id="1",
        question_text="Test?",
        golden_answer="Meets",
        predicted_answer="Does Not Meet",
    )
    assert qe_false.is_correct is False

    # Test case insensitivity
    qe_case = QuestionEvaluation(
        question_id="1",
        question_text="Test?",
        golden_answer="meets",
        predicted_answer="MEETS ",
    )
    assert qe_case.is_correct is True


def test_conversation_evaluator_metrics():
    """Test metric calculation in ConversationEvaluator, including Cohen's Kappa."""
    evals = [
        QuestionEvaluation("1", "Q1", "yes", "yes"),
        QuestionEvaluation("2", "Q2", "yes", "no"),
        QuestionEvaluation("3", "Q3", "no", "no"),
        QuestionEvaluation("4", "Q4", "no", "yes"),
    ]
    # Matrix:
    # Gold: Yes, Yes, No, No
    # Pred: Yes, No, No, Yes
    # Agreement: 2/4 (50%)

    evaluator = ConversationEvaluator("conv_1", evals, valid_labels=["yes", "no"])
    metrics = evaluator.compute_metrics()

    assert metrics["accuracy"] == 0.5
    assert metrics["total_questions"] == 4
    # Kappa should be 0 because observed agreement (0.5) equals random chance agreement (0.5)
    # P(Yes) = 0.5, P(No) = 0.5 for both raters
    assert metrics["cohens_kappa"] == 0.0


def test_conversation_evaluator_perfect_agreement():
    """Test Cohen's Kappa for perfect agreement."""
    evals = [
        QuestionEvaluation("1", "Q1", "yes", "yes"),
        QuestionEvaluation("2", "Q2", "no", "no"),
    ]
    evaluator = ConversationEvaluator("conv_1", evals, valid_labels=["yes", "no"])
    metrics = evaluator.compute_metrics()

    assert metrics["accuracy"] == 1.0
    assert metrics["cohens_kappa"] == 1.0


def test_run_automated_evaluation_join():
    """Test the joining and processing logic of run_automated_evaluation."""
    results_df = pd.DataFrame(
        [
            {
                "resource_id": "c1",
                "question_id": "q1",
                "answer": "meets",
                "question_text": "Q1",
            },
            {
                "resource_id": "c1",
                "question_id": "q2",
                "answer": "meets",
                "question_text": "Q2",
            },
        ]
    )
    golden_df = pd.DataFrame(
        [
            {"resource_id": "c1", "question_id": "q1", "answer": "meets"},
            {"resource_id": "c1", "question_id": "q2", "answer": "does not meet"},
        ]
    )

    reporter = run_automated_evaluation(results_df, golden_df, join_key="resource_id")
    summary = reporter.get_summary_dataframe()

    assert len(summary) == 1
    assert summary.iloc[0]["accuracy"] == 0.5


@patch("src.workflow.insight_refinements.analysis.get_oauth_token")
@patch("requests.get")
def test_analyzer_list_conversations(mock_get, mock_token, mock_config):
    """Test listing conversations with mocked API response."""
    mock_token.return_value = "fake-token"
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "conversations": [
            {"name": "projects/p/locations/l/conversations/c1"},
            {"name": "projects/p/locations/l/conversations/c2"},
        ],
        "nextPageToken": None,
    }

    analyzer = ConversationAnalyzer(mock_config)
    ids = analyzer.list_conversation_ids()

    assert ids == ["c1", "c2"]
    mock_get.assert_called_once()


@patch("src.workflow.insight_refinements.analysis.get_bq_client")
def test_bq_operator_load(mock_get_bq_client, mock_config):
    """Test BigQueryOperator load_dataframe_to_staging."""
    mock_client = mock_get_bq_client.return_value
    operator = BigQueryOperator(mock_config)
    df = pd.DataFrame({"col": [1, 2]})

    operator.load_dataframe_to_staging(df, "staging_table")

    # Verify the mock setup was called
    mock_client.load_table_from_dataframe.assert_called_once()
    # Verify job.result() was called
    mock_client.load_table_from_dataframe.return_value.result.assert_called_once()


@patch("src.workflow.insight_refinements.analysis.get_bq_client")
def test_bq_operator_merge(mock_get_bq_client, mock_config):
    """Test BigQueryOperator merge_staging_to_main."""
    mock_client = mock_get_bq_client.return_value
    operator = BigQueryOperator(mock_config)

    # Configure the mock to return an integer for num_dml_affected_rows
    mock_query_job = mock_client.query.return_value
    mock_query_job.num_dml_affected_rows = 10

    operator.merge_staging_to_main(
        staging_table="staging",
        main_table="main",
        merge_keys=["id"],
        update_columns=["val"],
    )

    mock_client.query.assert_called_once()
    mock_query_job.result.assert_called_once()

    # Check that the SQL contains the MERGE statement
    args, _ = mock_client.query.call_args
    assert "MERGE" in args[0]
    assert "test-project.test_dataset.main" in args[0]


@patch("src.workflow.insight_refinements.analysis.ConversationAnalyzer")
def test_run_batch_analysis_bulk(mock_analyzer_cls, mock_config):
    """Test run_batch_analysis with enable_bulk_analysis=True."""
    mock_config.analysis.enable_bulk_analysis = True
    mock_config.analysis.filter = "test-filter"
    mock_config.analysis.analysis_percentage = 50

    mock_instance = mock_analyzer_cls.return_value
    mock_instance.bulk_analyze_conversations.return_value = {"name": "op123"}
    mock_instance.wait_for_operation.return_value = {"status": "done"}

    results = run_batch_analysis(mock_config)

    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["operation_metadata"] == {"status": "done"}

    mock_instance.bulk_analyze_conversations.assert_called_once()
    mock_instance.wait_for_operation.assert_called_once_with(
        {"name": "op123"}, timeout_seconds=3600
    )
    mock_instance.list_conversation_ids.assert_not_called()
