import datetime  # Need datetime for timestamp
import json  # Need json for CSV serialization
import logging
import uuid  # Need uuid for unique id
from typing import Optional, Union

# Third-party imports
# import pandas as pd # Removed unused import (Pandas is used via KnowledgeBase/TabularStore)
from mcp.server.fastmcp import Context
from mcp.types import INVALID_REQUEST
from pydantic import ValidationError as PydanticValidationError

# Local application imports
# Import the central instances registry
from ..core import instances

# Import custom exceptions and error models
from ..core.exceptions import AYAServerError
from ..core.models import research_models
from ..core.models.enums import DirectiveType, OverallStatus
from ..core.models.error_models import OperationalErrorData
from ..core.models.research_models import (
    DefineTaskArgs,
    DefineTaskResult,
    ExportFormat,
    ExportResult,
    ExportResultsArgs,
    InstructionObjectV3,
    StrategicReviewDirective,
    SubmitClarificationResult,
    SubmitInquiryReportArgs,
    SubmitUserClarificationArgs,
)
from ..storage.tabular_store import INTERNAL_COLS
from ..utils.artifact_manager import get_artifact_path  # ADDED IMPORT

# Import the central error handler
from .error_handler import handle_aya_exception

logger = logging.getLogger(__name__)


async def handle_define_task(args: DefineTaskArgs, ctx: Context) -> DefineTaskResult:
    """
    Defines the research scope and parameters. Requires AWAITING_TASK_DEFINITION state.
    Input: DefineTaskArgs (containing TaskDefinition).
    Output: DefineTaskResult, which includes confirmation and the *first* directive object
            (InstructionObjectV3 or StrategicReviewDirective with embedded context)
            within the 'instruction_payload' field, or a status update.
    """
    operation = "handle_define_task"
    logger.info(f"Tool '{operation}' called.")
    # Access components via the instances registry
    state_manager = instances.state_manager
    planner = instances.planner
    directive_builder = instances.directive_builder

    try:
        if not state_manager or not planner or not directive_builder:
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details="Required server components not initialized.",
            )
            raise AYAServerError(
                "Server configuration error: Core components missing.", data=op_data
            )

        if state_manager.status != OverallStatus.AWAITING_TASK_DEFINITION:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details=f"Invalid state: {state_manager.status.value}",
            )
            raise AYAServerError(
                f"Cannot define task in current state: {state_manager.status.value}",
                code=INVALID_REQUEST,
                data=op_data,
            )

        await state_manager.define_task(args.task_definition)
        planner.set_task_definition(args.task_definition)
        logger.info(
            f"Task defined successfully. New status: {state_manager.status.value}"
        )

        logger.debug("Planning the first directive after task definition.")
        next_directive: Optional[
            Union[InstructionObjectV3, StrategicReviewDirective]
        ] = None
        result_status: research_models.RequestDirectiveResultStatus
        task_def = args.task_definition
        planner_signal_tuple = planner.determine_next_directive()

        if planner_signal_tuple == "CLARIFICATION_NEEDED":
            clarification_reason = "Planner requires immediate clarification based on task definition/seeds."
            await state_manager.request_clarification(clarification_reason)
            next_directive = InstructionObjectV3(
                research_goal_context=task_def.task_description,
                inquiry_focus=f"User Clarification Needed: {clarification_reason}",
                focus_areas=[
                    "Please provide guidance via 'research/submit_user_clarification'."
                ],
                allowed_tools=["research/submit_user_clarification"],
                directive_type=DirectiveType.CLARIFICATION,
            )
            result_status = (
                research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
            )
        elif planner_signal_tuple is None:
            await state_manager.complete_research()
            next_directive = InstructionObjectV3(
                research_goal_context=task_def.task_description,
                inquiry_focus="Research Complete, you should export or preview results.",
                allowed_tools=[
                    "research/preview_results",
                    "research/export_results",
                ],
                directive_type=DirectiveType.COMPLETION,
            )
            result_status = (
                research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
            )
        else:
            next_directive = directive_builder.build_directive(
                planner_signal_tuple, task_def
            )
            await state_manager.directive_issued(next_directive)
            directive_id = getattr(
                next_directive,
                "instruction_id",
                getattr(next_directive, "directive_id", "N/A"),
            )
            logger.info(f"Issuing first instruction {directive_id}")
            result_status = (
                research_models.RequestDirectiveResultStatus.DIRECTIVE_ISSUED
            )

        if not next_directive:
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details="Failed to generate initial instruction after planning.",
            )
            raise AYAServerError(
                "Internal error: Failed to generate initial instruction.",
                data=op_data,
            )

        summary_text = (
            f"Next step: {getattr(next_directive, 'inquiry_focus', 'Unknown')}"
        )

        return DefineTaskResult(
            message=f"Task '{args.task_definition.task_description[:30]}...' defined successfully.",
            status=result_status,
            summary=summary_text,
            instruction_payload=next_directive,
            error_message=None,
        )
    except (AYAServerError, PydanticValidationError, Exception) as e:
        # Raise the converted McpError, preserving the original exception context
        raise handle_aya_exception(e, operation) from e


async def handle_submit_inquiry_report(
    args: SubmitInquiryReportArgs, ctx: Context
) -> research_models.SubmitReportAndGetDirectiveResult:
    """
    Submits results for the active directive (standard or strategic review). Requires CONDUCTING_INQUIRY state.
    Input: SubmitInquiryReportArgs (containing InquiryReport with findings OR strategic_decision).
    Output: SubmitReportAndGetDirectiveResult, which includes confirmation and the *next* directive object
            (InstructionObjectV3 or StrategicReviewDirective with embedded context)
            within the 'instruction_payload' field, or a status update (completion/clarification).
    """
    operation = "handle_submit_inquiry_report"
    logger.info(f"Tool '{operation}' called.")
    # Access components via the instances registry
    state_manager = instances.state_manager
    report_handler = instances.report_handler
    planner = instances.planner
    directive_builder = instances.directive_builder

    try:
        if (
            not state_manager
            or not report_handler
            or not planner
            or not directive_builder
        ):
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details="Required server components not initialized.",
            )
            raise AYAServerError(
                "Server configuration error: Core components missing.", data=op_data
            )

        if state_manager.status != OverallStatus.CONDUCTING_INQUIRY:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details=f"Invalid state: {state_manager.status.value}",
            )
            raise AYAServerError(
                f"Cannot submit report in current state: {state_manager.status.value}. Waiting for directive?",
                code=INVALID_REQUEST,
                data=op_data,
            )

        active_directive = state_manager.active_instruction
        if not active_directive:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details="No active directive found.",
            )
            raise AYAServerError(
                "Inconsistent state: No active directive found.", data=op_data
            )

        report_outcome = await report_handler.process_report(
            args.inquiry_report, active_directive
        )
        resulting_status, strategic_decision, strategic_targets = report_outcome

        if resulting_status is None:
            op_data = OperationalErrorData(
                component="ReportHandler",
                operation="process_report",
                details="process_report returned None status unexpectedly.",
            )
            raise AYAServerError(
                "Internal server error processing report outcome signal.",
                data=op_data,
            )

        next_directive_payload: Optional[
            Union[InstructionObjectV3, StrategicReviewDirective]
        ] = None
        result_status: research_models.RequestDirectiveResultStatus
        planner_signal = None
        task_def = state_manager.task_definition

        if not task_def:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details="TaskDefinition missing.",
            )
            raise AYAServerError(
                "Server state inconsistent: TaskDefinition missing.", data=op_data
            )

        if strategic_decision:
            logger.info(f"Processing client strategic decision: {strategic_decision}")
            # Prepare client assessment dictionary to pass to the planner
            client_assessment = {
                "discovery_needed": strategic_decision == "DISCOVER",
                "enrichment_needed": strategic_decision
                in ["ENRICH", "ENRICH_SPECIFIC"],
                "prioritized_enrichment_targets": (
                    strategic_targets
                    if strategic_decision == "ENRICH_SPECIFIC"
                    else None
                ),
                "request_user_clarification": strategic_decision == "CLARIFY_USER",
                "goal_achieved": strategic_decision == "FINALIZE",
            }
            # Check if the client explicitly requested clarification
            if strategic_decision == "CLARIFY_USER":
                # Directly transition state without calling planner
                await state_manager.request_clarification(
                    "Client requested user clarification via strategic decision."
                )
                result_status = (
                    research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                )
                # No directive payload needed when clarification is requested
                next_directive_payload = None
            else:
                # For other decisions (ENRICH, DISCOVER, FINALIZE, etc.), call the planner
                planner_signal_tuple = planner.determine_next_directive(
                    client_assessment=client_assessment
                )
                # Process the planner's signal based on the client's decision
                if planner_signal_tuple == "CLARIFICATION_NEEDED":
                    # Planner determined clarification is needed even after client decision
                    await state_manager.request_clarification(
                        "Planner requires clarification based on strategic decision outcome."
                    )
                    result_status = (
                        research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                    )
                elif planner_signal_tuple is None:
                    # Planner determined completion
                    await state_manager.complete_research()
                    result_status = (
                        research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                    )
                    # Create completion directive payload for consistency
                    next_directive_payload = InstructionObjectV3(
                        research_goal_context=task_def.task_description,
                        inquiry_focus="Research Complete. Please export or preview the results.",
                        allowed_tools=[
                            "research/preview_results",
                            "research/export_results",
                        ],
                        directive_type=DirectiveType.COMPLETION,
                    )
                else:
                    # Planner returned a directive signal (ENRICH, DISCOVERY, NEEDS_STRATEGIC_REVIEW)
                    planner_signal = planner_signal_tuple
                    # We will build the directive payload later based on this signal
                    result_status = (
                        research_models.RequestDirectiveResultStatus.DIRECTIVE_ISSUED  # Assume directive will be issued
                    )
                # result_status will be set later when the directive is built

        elif resulting_status == OverallStatus.AWAITING_DIRECTIVE:
            logger.debug("Report processed, planning next directive.")
            planner_signal_tuple = planner.determine_next_directive(
                client_assessment=None  # Pass assessment if available from report
            )
            if planner_signal_tuple == "CLARIFICATION_NEEDED":
                # Trigger strategic review if clarification needed after standard report
                planner_signal = (
                    "NEEDS_STRATEGIC_REVIEW",
                    "critical_obstacles",
                )  # Example reason
            elif planner_signal_tuple is None:
                await state_manager.complete_research()
                result_status = (
                    research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                )
                next_directive_payload = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus="Research Complete. Please export or preview the results.",
                    allowed_tools=[
                        "research/preview_results",
                        "research/export_results",
                    ],
                    directive_type=DirectiveType.COMPLETION,  # Placeholder type
                )
            else:
                planner_signal = planner_signal_tuple
        else:
            logger.info(
                f"Report processing resulted in state {resulting_status.value}. No immediate planning needed."
            )
            if resulting_status == OverallStatus.AWAITING_USER_CLARIFICATION:
                result_status = (
                    research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                )
            elif resulting_status == OverallStatus.FATAL_ERROR:
                op_data = OperationalErrorData(
                    component="StateManager",
                    operation="report_received",
                    details="Server entered FATAL_ERROR state.",
                )
                raise AYAServerError(
                    "Server entered FATAL_ERROR state during report processing.",
                    data=op_data,
                )
            else:  # Handle unexpected states
                op_data = OperationalErrorData(
                    component="StateManager",
                    operation="report_received",
                    details=f"Unexpected state {resulting_status.value} after report processing.",
                )
                raise AYAServerError(
                    f"Unexpected state after report: {resulting_status.value}",
                    data=op_data,
                )

        # Build directive only if planner_signal is set
        if planner_signal:
            next_directive_payload = directive_builder.build_directive(
                planner_signal, task_def
            )
            await state_manager.directive_issued(next_directive_payload)
            result_status = (
                research_models.RequestDirectiveResultStatus.DIRECTIVE_ISSUED
            )
            logger.info(
                f"Issuing next directive: {getattr(next_directive_payload, 'instruction_id', getattr(next_directive_payload, 'directive_id', 'N/A'))}"
            )

        # Determine summary text based on the final result_status
        summary_text = "Status Update"
        if (
            result_status
            == research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
        ):
            summary_text = "Research Complete. Results available for preview or export."
        elif (
            result_status
            == research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
        ):
            summary_text = "User clarification needed."
        elif next_directive_payload:
            summary_text = f"Next step: {getattr(next_directive_payload, 'inquiry_focus', 'Unknown')}"

        logger.info(
            f"RETURN_DEBUG: Final next_directive_payload before return: {next_directive_payload.model_dump_json(indent=2) if next_directive_payload else 'None'}"
        )
        logger.info(
            f"FINAL RETURN STATE: result_status={result_status}, has_payload={next_directive_payload is not None}"
        )
        if next_directive_payload:
            logger.debug(
                f"FINAL RETURN PAYLOAD: {next_directive_payload.model_dump_json(indent=2)}"
            )
        else:
            logger.debug("FINAL RETURN PAYLOAD: None")

        return research_models.SubmitReportAndGetDirectiveResult(
            message="Report submitted successfully.",
            status=result_status,
            summary=summary_text,
            instruction_payload=next_directive_payload,
            error_message=None,
        )
    except (AYAServerError, PydanticValidationError, Exception) as e:
        raise handle_aya_exception(e, operation) from e


async def handle_submit_user_clarification(
    args: SubmitUserClarificationArgs, ctx: Context
) -> SubmitClarificationResult:
    """
    Provides user input when clarification is requested. Requires AWAITING_USER_CLARIFICATION state.
    Input: SubmitUserClarificationArgs (containing clarification text).
    Output: SubmitClarificationResult, which includes confirmation and the *next* directive object
            (InstructionObjectV3 or StrategicReviewDirective with embedded context)
            within the 'instruction_payload' field, or a status update. Also includes the new server status and available tools.
    """
    operation = "handle_submit_user_clarification"
    logger.info(f"Tool '{operation}' called.")
    # Access components via the instances registry
    state_manager = instances.state_manager
    planner = instances.planner
    directive_builder = instances.directive_builder

    try:
        if not state_manager or not planner or not directive_builder:
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details="Required server components not initialized.",
            )
            raise AYAServerError(
                "Server configuration error: Core components missing.", data=op_data
            )

        if state_manager.status != OverallStatus.AWAITING_USER_CLARIFICATION:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details=f"Invalid state: {state_manager.status.value}",
            )
            raise AYAServerError(
                f"Cannot submit clarification in current state: {state_manager.status.value}",
                code=INVALID_REQUEST,
                data=op_data,
            )

        # Process clarification (e.g., update KB, inform Planner)
        logger.debug(f"Processing clarification: '{args.clarification}'")
        try:
            planner.incorporate_clarification(args.clarification)
            logger.debug("Planner incorporate_clarification completed.")
            await state_manager.clarification_received(args.clarification)
            logger.info(
                f"Clarification received and processed by StateManager. New status: {state_manager.status.value}"
            )
            # --- Check for Finalization Intent in Clarification ---
            # Simple check for now, could be more sophisticated
            if args.clarification and "finalize" in args.clarification.lower():
                logger.info(
                    "User clarification indicates finalization. Completing research."
                )
                await state_manager.complete_research()
                # Fall through to handle RESEARCH_COMPLETE state below
            elif state_manager.status != OverallStatus.AWAITING_DIRECTIVE:
                # If clarification didn't immediately resolve to AWAITING_DIRECTIVE
                # (e.g., stayed in AWAITING_USER_CLARIFICATION or went to ERROR),
                # handle appropriately. For now, assume it transitions or error handled.
                logger.warning(
                    f"State after clarification is {state_manager.status.value}, expected AWAITING_DIRECTIVE or RESEARCH_COMPLETE."
                )

        except Exception as proc_err:
            logger.error(f"Error processing clarification: {proc_err}", exc_info=True)
            op_data = OperationalErrorData(
                component="Planner/StateManager",
                operation="process_clarification",
                details=str(proc_err),
            )
            raise AYAServerError(
                f"Failed to process clarification: {proc_err}", data=op_data
            ) from proc_err

        # --- Plan Next Step After Clarification ---
        next_directive_payload: Optional[
            Union[InstructionObjectV3, StrategicReviewDirective]
        ] = None
        result_status: research_models.RequestDirectiveResultStatus
        planner_signal = None
        task_def = state_manager.task_definition

        if not task_def:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details="TaskDefinition missing after clarification.",
            )
            raise AYAServerError(
                "Server state inconsistent: TaskDefinition missing.", data=op_data
            )

        if state_manager.status == OverallStatus.RESEARCH_COMPLETE:
            result_status = (
                research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
            )
            next_directive_payload = InstructionObjectV3(
                research_goal_context=task_def.task_description,
                inquiry_focus="Research Complete (finalized via clarification). Please export or preview the results.",
                allowed_tools=[
                    "research/preview_results",
                    "research/export_results",
                ],
                directive_type=DirectiveType.COMPLETION,
            )
        elif state_manager.status == OverallStatus.AWAITING_DIRECTIVE:
            logger.debug("Planning next directive after clarification.")
            planner_signal_tuple = planner.determine_next_directive()

            if planner_signal_tuple == "CLARIFICATION_NEEDED":
                # This shouldn't ideally happen right after clarification, but handle it
                clarification_reason = (
                    "Planner still requires clarification after user input."
                )
                await state_manager.request_clarification(clarification_reason)
                next_directive_payload = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus=f"User Clarification Still Needed: {clarification_reason}",
                    focus_areas=[
                        "Please provide further guidance via 'research/submit_user_clarification'."
                    ],
                    allowed_tools=["research/submit_user_clarification"],
                    directive_type=DirectiveType.ENRICHMENT,  # Placeholder
                )
                result_status = (
                    research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                )
            elif planner_signal_tuple is None:
                # Planner decided to complete after clarification
                await state_manager.complete_research()
                result_status = (
                    research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                )
                next_directive_payload = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus="Research Complete (decided after clarification). Please export or preview the results.",
                    allowed_tools=[
                        "research/preview_results",
                        "research/export_results",
                    ],
                    directive_type=DirectiveType.COMPLETION,
                )
            else:
                # Build the next directive
                planner_signal = planner_signal_tuple
                next_directive_payload = directive_builder.build_directive(
                    planner_signal, task_def
                )
                await state_manager.directive_issued(next_directive_payload)
                result_status = (
                    research_models.RequestDirectiveResultStatus.DIRECTIVE_ISSUED
                )
                logger.info(
                    f"Issuing next directive after clarification: {getattr(next_directive_payload, 'instruction_id', getattr(next_directive_payload, 'directive_id', 'N/A'))}"
                )
        else:
            # Handle unexpected states after clarification
            op_data = OperationalErrorData(
                component="StateManager",
                operation="clarification_received",
                details=f"Unexpected state {state_manager.status.value} after clarification.",
            )
            raise AYAServerError(
                f"Unexpected state after clarification: {state_manager.status.value}",
                data=op_data,
            )

        # Determine summary text
        summary_text = "Status Update"
        if (
            result_status
            == research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
        ):
            summary_text = "Research Complete. Results available."
        elif (
            result_status
            == research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
        ):
            summary_text = "Further user clarification needed."
        elif next_directive_payload:
            summary_text = f"Next step: {getattr(next_directive_payload, 'inquiry_focus', 'Unknown')}"

        # Get final status and tools for the result payload
        final_status_str = state_manager.status.value
        available_tools = state_manager.get_available_tools()

        logger.debug(
            f"Final status: {final_status_str}, Available tools: {available_tools}"
        )

        return SubmitClarificationResult(
            message="Clarification processed.",
            status=result_status,
            summary=summary_text,
            instruction_payload=next_directive_payload,
            error_message=None,
            new_status=final_status_str,
            available_tools=available_tools,
        )

    except (AYAServerError, PydanticValidationError, Exception) as e:
        raise handle_aya_exception(e, operation) from e


async def handle_preview_results(args: dict, ctx: Context) -> str:
    """
    Returns a preview of the current knowledge base as a formatted string (e.g., markdown table).
    Requires RESEARCH_COMPLETE state.
    Input: Empty args dict (for now).
    Output: Formatted string preview.
    """
    operation = "handle_preview_results"
    logger.info(f"Tool '{operation}' called.")
    state_manager = instances.state_manager
    kb = instances.knowledge_base

    try:
        if not state_manager or not kb:
            raise AYAServerError("Server components not initialized.")
        if state_manager.status != OverallStatus.RESEARCH_COMPLETE:
            raise AYAServerError(
                f"Cannot preview results in state: {state_manager.status.value}",
                code=INVALID_REQUEST,
            )

        preview_df = kb.get_data_preview(num_rows=20)  # Get more rows for preview
        if preview_df.empty:
            return "Knowledge base is empty. No results to preview."

        # Use Pandas to format as markdown
        # Reset index before converting to markdown to include index columns
        # FIX: Directly convert to markdown, including the index ('Country')
        markdown_preview = preview_df.to_markdown()
        return (
            f"**Results Preview (First {len(preview_df)} rows):**\n\n{markdown_preview}"
        )

    except (AYAServerError, PydanticValidationError, Exception) as e:
        raise handle_aya_exception(e, operation) from e


async def handle_export_results(args: ExportResultsArgs, ctx: Context) -> ExportResult:
    """
    Exports the full knowledge base to a specified file format (CSV, JSON, Parquet).
    Requires RESEARCH_COMPLETE state.
    Input: ExportResultsArgs (specifying format).
    Output: ExportResult (containing server-side file path).
    """
    operation = "handle_export_results"
    logger.info(f"Tool '{operation}' called with format: {args.format.value}")
    state_manager = instances.state_manager
    kb = instances.knowledge_base
    # artifact_manager = instances.artifact_manager # REMOVED - Incorrect usage

    try:
        # Check required components (artifact_manager removed)
        if not state_manager or not kb:
            raise AYAServerError(
                "Core server components (StateManager, KnowledgeBase) not initialized."
            )
        if state_manager.status != OverallStatus.RESEARCH_COMPLETE:
            raise AYAServerError(
                f"Cannot export results in state: {state_manager.status.value}",
                code=INVALID_REQUEST,
            )

        # Use get_artifact_path utility
        # Note: os.makedirs is handled within get_artifact_path if filename is provided
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        file_format = args.format.value
        filename = f"export_{timestamp}_{unique_id}.{file_format}"

        try:
            # Get the full path, creating directories as needed
            full_file_path = get_artifact_path(filename=filename)
            logger.debug(f"Generated export file path: {full_file_path}")
        except (OSError, PermissionError, FileNotFoundError) as path_err:
            op_data = OperationalErrorData(
                component="ArtifactManagerUtil",
                operation="get_artifact_path",
                details=f"Failed to get or create artifact path for '{filename}': {path_err}",
            )
            raise AYAServerError(
                "Filesystem error: Could not determine or create export path.",
                data=op_data,
            ) from path_err

        # Fetch all data from the Knowledge Base
        logger.debug(
            f"Attempting to fetch all data from KB for export. KB instance: {kb}"
        )
        try:
            all_data_df = kb.get_full_dataset()
            logger.debug(
                f"KB get_full_dataset returned. DataFrame is empty: {all_data_df.empty}. Shape: {all_data_df.shape if not all_data_df.empty else 'N/A'}"
            )
        except Exception as fetch_err:
            logger.error(
                f"Error directly calling kb.get_full_dataset: {fetch_err}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation="get_full_dataset",
                details=f"Unexpected error fetching data: {type(fetch_err).__name__}: {fetch_err}",
            )
            raise AYAServerError(
                f"Failed to retrieve data from Knowledge Base for export. Error: {type(fetch_err).__name__}",
                data=op_data,
            ) from fetch_err

        if all_data_df.empty:
            logger.warning("Export requested, but Knowledge Base is empty.")
            # Create an empty file as requested

        # Prepare DataFrame for export (reset index)
        df_export = all_data_df  # Default to original df if empty
        if not all_data_df.empty:
            df_export = all_data_df.reset_index(
                drop=True
            )  # Flatten index into columns, DISCARDING the index
            logger.debug(
                f"DataFrame index reset for export. Columns: {df_export.columns.tolist()}"
            )

        # Write DataFrame to file based on format
        try:
            row_count = len(all_data_df)  # Get row count from original df before reset

            if args.format == ExportFormat.CSV:
                # Serialize complex object columns to JSON strings for CSV export
                # Operate on the df_export DataFrame
                logger.debug(
                    f"Serializing internal columns to JSON for CSV export: {INTERNAL_COLS}"
                )
                for col in INTERNAL_COLS:
                    if col in df_export.columns:
                        # Check if the column actually contains list/dict before applying serialization
                        # This check avoids errors if a column expected to be complex isn't, or is all null
                        needs_serialization = (
                            df_export[col]
                            .apply(lambda x: isinstance(x, (list, dict)))
                            .any()
                        )
                        if needs_serialization:
                            logger.debug(f"Serializing column '{col}' to JSON.")
                            # Apply json.dumps only to non-null values that are lists or dicts
                            df_export[col] = df_export[col].apply(
                                lambda x: (
                                    json.dumps(x) if isinstance(x, (list, dict)) else x
                                )
                            )
                # Export the modified DataFrame
                df_export.to_csv(
                    full_file_path, index=False
                )  # index=False as it's reset
            elif args.format == ExportFormat.JSON:
                # Use pandas to_json with 'records' orientation for list of objects
                # This handles NaNs correctly by default (outputs 'null')
                # Complex objects (lists/dicts) in internal columns are handled correctly by pandas
                # Ensure date formatting if needed (though not typical for this KB)
                df_export.to_json(  # Use df_export
                    full_file_path, orient="records", indent=4, date_format="iso"
                )
            elif args.format == ExportFormat.PARQUET:
                df_export.to_parquet(
                    full_file_path, index=False
                )  # Use df_export, index=False
            else:
                # Should be caught by Pydantic validation, but handle defensively
                raise ValueError(f"Unsupported export format: {args.format}")

            logger.info(f"Successfully exported {row_count} rows to {full_file_path}")
        except (OSError, PermissionError, ValueError) as write_err:
            op_data = OperationalErrorData(
                component="Filesystem/Pandas",
                operation=f"export_to_{args.format.value}",
                details=f"Failed to write export file '{full_file_path}': {write_err}",
            )
            raise AYAServerError(
                f"Filesystem/Export error: Could not write export file. {type(write_err).__name__}",
                data=op_data,
            ) from write_err
        except Exception as unexpected_write_err:
            op_data = OperationalErrorData(
                component="Filesystem/Pandas",
                operation=f"export_to_{args.format.value}",
                details=f"Unexpected error writing export file '{full_file_path}': {unexpected_write_err}",
            )
            raise AYAServerError(
                f"Unexpected error during export: {type(unexpected_write_err).__name__}",
                data=op_data,
            ) from unexpected_write_err

        return ExportResult(file_path=str(full_file_path))

    except (AYAServerError, PydanticValidationError, Exception) as e:
        # Log the original error before handling
        logger.error(f"Error during {operation}: {e}", exc_info=True)
        raise handle_aya_exception(e, operation) from e
