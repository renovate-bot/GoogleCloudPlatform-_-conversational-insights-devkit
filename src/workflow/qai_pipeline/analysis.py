# Copyright 2026 Google. This software is provided as-is, without warranty or representation for any use or purpose. Your use of it is subject to your agreement with Google.
"""
Batch Analysis Logic for QAI Optimization Framework.

This module provides classes and functions for performing batch analysis
on CCAI conversations and exporting the results to BigQuery.
"""

import json
import time
import requests
import google.auth
import google.auth.transport.requests
import pandas as pd
from typing import Any, Dict, List, Callable
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from src.workflow.qai_pipeline.schemas.input import BatchAnalysisInput
from src.workflow.qai_pipeline.utils import handle_api_quota, qai_logger, get_bq_client


def get_oauth_token() -> str:
    """
    Retrieves a Google OAuth2 token using Application Default Credentials (ADC).

    Returns:
        str: The OAuth2 access token.
    """
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token


def get_headers(token: str) -> Dict[str, str]:
    """
    Helper function to create HTTP headers with the provided token.

    Args:
        token (str): OAuth2 token.

    Returns:
        Dict[str, str]: Headers dictionary.
    """
    return {
        "charset": "utf-8",
        "Content-type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def refresh_token_if_unauthorized(func: Callable) -> Callable:
    """
    Decorator to refresh the OAuth token if a 401 Unauthorized error is received.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, "status_code") and e.response.status_code == 401:
                qai_logger.log("Token expired, refreshing...", severity="WARNING")
                self.token = get_oauth_token()
                # Retry the function once with new token
                return func(self, *args, **kwargs)
            raise e

    return wrapper


class ConversationAnalyzer:
    """
    Class for interacting with the CCAI Insights API to analyze conversations.
    """

    def __init__(self, config: BatchAnalysisInput):
        """
        Initializes the ConversationAnalyzer.

        Args:
            config (BatchAnalysisInput): The root configuration object.
        """
        self.config = config
        self.project_id = config.gcp.project_id
        self.location_id = config.gcp.location_id

        # Regional Endpoint Logic
        if config.ccai.insights_endpoint:
            self.insights_endpoint = config.ccai.insights_endpoint
        elif self.location_id == "us-central1":
            # us-central1 is often the global default endpoint
            self.insights_endpoint = "contactcenterinsights.googleapis.com"
        else:
            self.insights_endpoint = (
                f"{self.location_id}-contactcenterinsights.googleapis.com"
            )

        self.api_version = config.ccai.api_version
        self.token = get_oauth_token()

    @handle_api_quota()
    @refresh_token_if_unauthorized
    def list_conversation_ids(self, page_size: int = 1000) -> List[str]:
        """
        Lists conversation IDs within the configured CCAI project.

        Args:
            page_size (int): The number of conversations to retrieve per page.

        Returns:
            List[str]: A list of conversation IDs.
        """
        conversation_ids = []
        page_token = None
        base_url = f"https://{self.insights_endpoint}/{self.api_version}/projects/{self.project_id}/locations/{self.location_id}/conversations"

        while True:
            params = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            r = requests.get(base_url, headers=get_headers(self.token), params=params)
            r.raise_for_status()

            data = r.json()
            for conversation in data.get("conversations", []):
                conversation_id = conversation["name"].split("/")[-1]
                conversation_ids.append(conversation_id)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return conversation_ids

    @handle_api_quota()
    @refresh_token_if_unauthorized
    def start_analysis(
        self, conversation_id: str, analysis_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Triggers an analysis operation for a single conversation.

        Args:
            conversation_id (str): The ID of the conversation.
            analysis_config (Dict[str, Any]): The annotator configuration.

        Returns:
            Dict[str, Any]: The operation object returned by the API.
        """
        url = f"https://{self.insights_endpoint}/{self.api_version}/projects/{self.project_id}/locations/{self.location_id}/conversations/{conversation_id}/analyses"

        payload = {"annotatorSelector": analysis_config}
        r = requests.post(
            url, headers=get_headers(self.token), data=json.dumps(payload)
        )
        r.raise_for_status()

        return r.json()

    @handle_api_quota()
    @refresh_token_if_unauthorized
    def wait_for_operation(
        self, operation: Dict[str, Any], timeout_seconds: int = 900
    ) -> Dict[str, Any]:
        """
        Waits for a Long-Running Operation (LRO) to complete.

        Args:
            operation (Dict[str, Any]): The operation object.
            timeout_seconds (int): Maximum time to wait.

        Returns:
            Dict[str, Any]: The operation metadata upon completion.
        """
        operation_name = operation["name"]
        url = f"https://{self.insights_endpoint}/{self.api_version}/{operation_name}"

        start_time = time.time()
        while True:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"Operation {operation_name} timed out after {timeout_seconds}s"
                )

            r = requests.get(url, headers=get_headers(self.token))
            r.raise_for_status()
            result = r.json()

            if result.get("done"):
                if result.get("error"):
                    raise Exception(f"Operation failed: {result['error']}")
                return result.get("metadata", {})

            qai_logger.log(
                f"Operation {operation_name} still running...", severity="DEBUG"
            )
            time.sleep(10)

    @handle_api_quota()
    @refresh_token_if_unauthorized
    def bulk_analyze_conversations(self) -> Dict[str, Any]:
        """
        Triggers a bulk analysis operation for conversations matching the filter.

        Returns:
            Dict[str, Any]: The operation object returned by the API.
        """
        url = f"https://{self.insights_endpoint}/{self.api_version}/projects/{self.project_id}/locations/{self.location_id}/conversations:bulkAnalyze"

        payload = {
            "parent": f"projects/{self.project_id}/locations/{self.location_id}",
            "filter": self.config.analysis.filter,
            "analysisPercentage": self.config.analysis.analysis_percentage,
            "annotatorSelector": self.config.analysis.annotator_selector,
        }

        r = requests.post(
            url, headers=get_headers(self.token), data=json.dumps(payload)
        )
        r.raise_for_status()

        return r.json()


class BigQueryOperator:
    """
    Class for managing BigQuery operations for analysis results.
    """

    def __init__(self, config: BatchAnalysisInput):
        """
        Initializes the BigQueryOperator.

        Args:
            config (BatchAnalysisInput): The root configuration object.
        """
        self.project_id = config.bigquery.project_id or config.gcp.project_id
        self.dataset_id = config.bigquery.dataset_id
        self.client = get_bq_client(config)

    def load_dataframe_to_staging(
        self, df: pd.DataFrame, table_id: str, write_disposition: str = "WRITE_TRUNCATE"
    ):
        """
        Loads a Pandas DataFrame into a BigQuery staging table.

        Args:
            df (pd.DataFrame): The data to load.
            table_id (str): The staging table ID.
            write_disposition (str): The BQ write disposition (default: WRITE_TRUNCATE).
        """
        full_table_id = f"{self.project_id}.{self.dataset_id}.{table_id}"
        job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)

        job = self.client.load_table_from_dataframe(
            df, full_table_id, job_config=job_config
        )
        job.result()
        qai_logger.log(
            "Loaded rows into staging", table_id=full_table_id, row_count=len(df)
        )

    def merge_staging_to_main(
        self,
        staging_table: str,
        main_table: str,
        merge_keys: List[str],
        update_columns: List[str],
    ):
        """
        Merges data from a staging table into a main table.

        Args:
            staging_table (str): Staging table ID.
            main_table (str): Main table ID.
            merge_keys (List[str]): Columns to use for matching.
            update_columns (List[str]): Columns to update on match.
        """
        full_staging = f"{self.project_id}.{self.dataset_id}.{staging_table}"
        full_main = f"{self.project_id}.{self.dataset_id}.{main_table}"

        on_clause = " AND ".join([f"T.{k} = S.{k}" for k in merge_keys])
        update_set = ", ".join([f"{c} = S.{c}" for c in update_columns])
        insert_cols = ", ".join(update_columns + merge_keys)
        insert_vals = ", ".join([f"S.{c}" for c in update_columns + merge_keys])

        sql = f"""
        MERGE `{full_main}` T
        USING `{full_staging}` S
        ON {on_clause}
        WHEN MATCHED THEN
            UPDATE SET {update_set}
        WHEN NOT MATCHED BY TARGET THEN
            INSERT ({insert_cols})
            VALUES ({insert_vals})
        """

        query_job = self.client.query(sql)
        query_job.result()
        qai_logger.log(
            "BigQuery Merge complete", affected_rows=query_job.num_dml_affected_rows
        )


def run_batch_analysis(config: BatchAnalysisInput):
    """
    High-level orchestrator for the batch analysis workflow.

    Args:
        config (BatchAnalysisInput): The validated configuration.
    """
    analyzer = ConversationAnalyzer(config)

    if config.analysis.enable_bulk_analysis:
        qai_logger.log(
            "Starting Bulk Analysis...",
            filter=config.analysis.filter,
            percentage=config.analysis.analysis_percentage,
        )
        try:
            op = analyzer.bulk_analyze_conversations()
            qai_logger.log(
                "Bulk Analysis Operation triggered", operation_name=op.get("name")
            )

            # Optionally wait for the operation, but since it's a bulk op, it might take a long time.
            # For now, let's wait to be consistent with the other flow,
            # assuming the user runs this in a background job.
            metadata = analyzer.wait_for_operation(
                op, timeout_seconds=3600
            )  # Increased timeout for bulk

            # Bulk analysis results are not returned directly in the same way as single analysis.
            # We return a summary.
            return [{"status": "success", "operation_metadata": metadata}]
        except Exception as e:
            qai_logger.log(
                "Bulk Analysis failed",
                severity="ERROR",
                error=str(e),
            )
            return [{"status": "failed", "error": str(e)}]

    qai_logger.log("Listing conversations for batch job...")
    all_ids = analyzer.list_conversation_ids()

    total_to_analyze = int(len(all_ids) * (config.analysis.analysis_percentage / 100))
    target_ids = all_ids[:total_to_analyze]
    qai_logger.log(
        "Starting batch analysis",
        target_count=len(target_ids),
        total_available=len(all_ids),
    )

    results = []

    def process_id(cid):
        try:
            op = analyzer.start_analysis(cid, config.analysis.annotator_selector)
            analyzer.wait_for_operation(op)
            return {"conversation_id": cid, "status": "success"}
        except Exception as e:
            qai_logger.log(
                "Failed to analyze conversation",
                severity="ERROR",
                conversation_id=cid,
                error=str(e),
            )
            return {"conversation_id": cid, "status": "failed", "error": str(e)}

    # Use ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(
        max_workers=config.analysis.max_concurrent_calls
    ) as executor:
        futures = [executor.submit(process_id, cid) for cid in target_ids]
        for future in as_completed(futures):
            results.append(future.result())

    return results
