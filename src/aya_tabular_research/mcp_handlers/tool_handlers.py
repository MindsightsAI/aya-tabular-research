# Standard library imports
import datetime
import io
import json
import logging
import os
import uuid
from typing import Optional, Union

# Third-party imports
from mcp.server.fastmcp import Context
from mcp.types import INVALID_REQUEST
from pydantic import ValidationError as PydanticValidationError

# Local application imports
# Import the central instances registry
from ..core import instances

# Import custom exceptions and error models
from ..core.exceptions import AYAServerError
from ..core.models import research_models
from ..core.models.enums import OverallStatus
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
                directive_type="ENRICHMENT",  # Using ENRICHMENT as placeholder type for clarification request
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
                directive_type="ENRICHMENT",  # Using ENRICHMENT as placeholder type for completion
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
            logger.info(f"Acting on client strategic decision: {strategic_decision}")
            if strategic_decision == "FINALIZE":
                await state_manager.complete_research()
                result_status = (
                    research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                )
                if not task_def:
                    op_data = OperationalErrorData(
                        component="MCPInterface",
                        operation=operation,
                        details="TaskDefinition missing unexpectedly during completion.",
                    )
                    raise AYAServerError(
                        "Internal error: TaskDefinition missing during completion.",
                        data=op_data,
                    )

                next_directive_payload = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus="Research Complete. Please export or preview the results.",
                    allowed_tools=[
                        "research/preview_results",
                        "research/export_results",
                    ],
                    directive_type="COMPLETION",  # Using COMPLETION as placeholder type
                )
                logger.info(
                    f"FINALIZE_DEBUG: Set next_directive_payload after FINALIZE: {next_directive_payload.model_dump_json(indent=2) if next_directive_payload else 'None'}"
                )
            elif strategic_decision == "DISCOVER":
                discovery_payload = planner._plan_discovery()
                planner_signal = ("DISCOVERY", discovery_payload)
            elif strategic_decision == "ENRICH":
                enrichment_outcome = planner._plan_enrichment()
                if isinstance(enrichment_outcome, tuple):
                    planner_signal = ("ENRICH", enrichment_outcome)
                elif enrichment_outcome == "CLARIFICATION_NEEDED":
                    await state_manager.request_clarification(
                        "Planner requires clarification to proceed with enrichment."
                    )
                    result_status = (
                        research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                    )
                else:  # Assume completion if not tuple or clarification
                    await state_manager.complete_research()
                    result_status = (
                        research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                    )
            elif strategic_decision == "ENRICH_SPECIFIC":
                enrichment_outcome = planner._plan_enrichment(
                    priority_targets=strategic_targets or []
                )
                if isinstance(enrichment_outcome, tuple):
                    planner_signal = ("ENRICH", enrichment_outcome)
                elif enrichment_outcome == "CLARIFICATION_NEEDED":
                    await state_manager.request_clarification(
                        "Planner requires clarification for specific enrichment targets."
                    )
                    result_status = (
                        research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                    )
                else:  # Assume completion
                    await state_manager.complete_research()
                    result_status = (
                        research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                    )
            elif strategic_decision == "CLARIFY_USER":
                await state_manager.request_clarification(
                    "Client requested user clarification during strategic review."
                )
                result_status = (
                    research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                )

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
                if not task_def:
                    op_data = OperationalErrorData(
                        component="MCPInterface",
                        operation=operation,
                        details="TaskDefinition missing unexpectedly during completion.",
                    )
                    raise AYAServerError(
                        "Internal error: TaskDefinition missing during completion.",
                        data=op_data,
                    )

                next_directive_payload = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus="Research Complete. Please export or preview the results.",
                    allowed_tools=[
                        "research/preview_results",
                        "research/export_results",
                    ],
                    directive_type="COMPLETION",  # Placeholder type
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
            # Simple check for now, could be more robust
            user_wants_to_finalize = (
                "finalize" in args.clarification.lower()
                or "stop" in args.clarification.lower()
            )
            # Get Task Definition - moved inside try block
            task_def = state_manager.task_definition
            if not task_def:
                op_data = OperationalErrorData(
                    component="StateManager",
                    operation=operation,
                    details="TaskDefinition missing after clarification processing.",
                )
                raise AYAServerError(
                    "Server state inconsistent: TaskDefinition missing.", data=op_data
                )

        except Exception as clar_err:
            logger.error(
                f"Error during clarification processing: {clar_err}", exc_info=True
            )
            raise  # Re-raise to be handled by main handler

        # --- Determine Next Step ---
        next_directive_payload: Optional[
            Union[InstructionObjectV3, StrategicReviewDirective]
        ] = None
        result_status: research_models.RequestDirectiveResultStatus

        # task_def is now assigned and checked within the try block above

        if user_wants_to_finalize:
            logger.info(
                "User clarification indicates finalization. Completing research."
            )
            await state_manager.complete_research()
            result_status = (
                research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
            )
            next_directive_payload = InstructionObjectV3(
                research_goal_context=task_def.task_description,
                inquiry_focus="Research Finalized by User. Please export or preview the results.",
                allowed_tools=["research/preview_results", "research/export_results"],
                directive_type="COMPLETION",
            )
        else:
            # Determine next step using the planner if not finalizing
            logger.debug("Planning next directive after clarification.")
            # task_def is already defined and checked above

            logger.debug("Calling planner.determine_next_directive()")
            planner_signal_tuple = planner.determine_next_directive()
            logger.debug(
                f"Planner determine_next_directive returned: {planner_signal_tuple}"
            )

            if planner_signal_tuple == "CLARIFICATION_NEEDED":
                clarification_reason = "Further clarification needed after user input."
                logger.warning(
                    f"Planner still needs clarification: {clarification_reason}"
                )
                await state_manager.request_clarification(clarification_reason)
                logger.debug("StateManager request_clarification completed.")
                result_status = (
                    research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                )
                # Build a minimal directive payload explaining this
                next_directive_payload = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus=f"Further Clarification Needed: {clarification_reason}",
                    focus_areas=[
                        "Please provide more specific guidance via 'research/submit_user_clarification'."
                    ],
                    allowed_tools=["research/submit_user_clarification"],
                    directive_type="ENRICHMENT",  # Placeholder type
                )

            elif planner_signal_tuple is None:
                logger.info(
                    "Planner indicated research is complete after clarification."
                )
                await state_manager.complete_research()
                logger.debug("StateManager complete_research completed.")
                result_status = (
                    research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                )
                # No specific payload needed here, handled by summary text logic later
            else:
                # Planner provided a directive signal (ENRICH, DISCOVERY, NEEDS_STRATEGIC_REVIEW)
                planner_signal = planner_signal_tuple
                logger.debug(f"Building directive for planner signal: {planner_signal}")
                next_directive_payload = directive_builder.build_directive(
                    planner_signal, task_def
                )
                await state_manager.directive_issued(next_directive_payload)
                logger.debug("StateManager directive_issued completed.")
                result_status = (
                    research_models.RequestDirectiveResultStatus.DIRECTIVE_ISSUED
                )
                logger.info(
                    f"Issuing next directive after clarification: {getattr(next_directive_payload, 'instruction_id', getattr(next_directive_payload, 'directive_id', 'N/A'))}"
                )

        summary_text = "Status Update"
        if (
            result_status
            == research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
        ):
            summary_text = "Research Complete."
        elif (
            result_status
            == research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
        ):
            summary_text = "Further user clarification needed."
        elif next_directive_payload:
            summary_text = f"Next step: {getattr(next_directive_payload, 'inquiry_focus', 'Unknown')}"

        # Get current status and tools after processing
        logger.debug("Retrieving final status and available tools...")
        current_status_str = state_manager.status.value
        current_status = state_manager.status  # Get the status enum object
        available_tools = state_manager.get_tools_for_state(current_status)
        logger.debug(
            f"Final status: {current_status_str}, Available tools: {available_tools}"
        )

        logger.debug("Constructing SubmitClarificationResult...")
        try:
            result = SubmitClarificationResult(
                message="Clarification submitted successfully.",
                status=result_status,
                summary=summary_text,
                instruction_payload=next_directive_payload,
                error_message=None,
                new_status=current_status_str,
                available_tools=available_tools,
            )
            logger.debug("SubmitClarificationResult constructed successfully.")
            return result
        except Exception as result_err:
            logger.error(
                f"Error constructing SubmitClarificationResult: {result_err}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details=f"Failed to construct result: {result_err}",
            )
            raise AYAServerError(
                "Internal error preparing clarification result.", data=op_data
            ) from result_err
    except (AYAServerError, PydanticValidationError, Exception) as e:
        raise handle_aya_exception(e, operation) from e


async def handle_preview_results(args: dict, ctx: Context) -> str:
    """
    Provides a preview (e.g., first 10 rows) of the final results as a CSV string.
    Requires RESEARCH_COMPLETE state.
    Input: Empty dictionary (args ignored).
    Output: String containing the CSV preview data.
    """
    operation = "handle_preview_results"
    logger.info(f"Tool '{operation}' called.")
    # Access components via the instances registry
    state_manager = instances.state_manager
    kb = instances.knowledge_base

    try:
        if not state_manager or not kb:
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details="Required server components not initialized.",
            )
            raise AYAServerError(
                "Server configuration error: Core components missing.", data=op_data
            )

        # Allow preview also in AWAITING_USER_CLARIFICATION state as per docs/roadmap
        allowed_states = [
            OverallStatus.RESEARCH_COMPLETE,
            OverallStatus.AWAITING_USER_CLARIFICATION,
        ]
        if state_manager.status not in allowed_states:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details=f"Invalid state: {state_manager.status.value}",
            )
            raise AYAServerError(
                f"Cannot preview results in current state: {state_manager.status.value}. Requires RESEARCH_COMPLETE or AWAITING_USER_CLARIFICATION.",
                code=INVALID_REQUEST,
                data=op_data,
            )

        # Fetch preview data
        logger.debug(
            f"Attempting to fetch preview data (max 10 rows) from KB. KB instance: {kb}"
        )
        try:
            preview_df = kb.get_data_preview(num_rows=10)
            logger.debug(
                f"KB get_data_preview returned. DataFrame is empty: {preview_df.empty}. Shape: {preview_df.shape if not preview_df.empty else 'N/A'}"
            )
            if not preview_df.empty:
                # Log DataFrame structure and types for debugging CSV conversion issues
                buffer = io.StringIO()
                preview_df.info(buf=buffer, verbose=True)
                df_info_str = buffer.getvalue()
                logger.debug(f"Preview DataFrame info:\n{df_info_str}")
        except Exception as preview_err:
            logger.error(
                f"Error directly calling kb.get_data_preview: {preview_err}",
                exc_info=True,
            )
            raise

        if preview_df.empty:
            logger.warning("Preview requested, but Knowledge Base is empty.")
            return "No data available for preview."

        # Convert preview DataFrame to CSV string
        try:
            csv_buffer = io.StringIO()
            preview_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            csv_buffer.close()
        except Exception as csv_err:
            logger.error(
                f"Error converting preview DataFrame to CSV: {csv_err}",
                exc_info=True,
            )
            raise AYAServerError(
                "Failed to generate CSV preview from data.",
                data={"error": str(csv_err)},
            ) from csv_err

        logger.info(
            f"Successfully generated preview CSV string (length: {len(csv_string)})."
        )
        return csv_string

    except (AYAServerError, PydanticValidationError, Exception) as e:
        raise handle_aya_exception(e, operation) from e


async def handle_export_results(args: ExportResultsArgs, ctx: Context) -> ExportResult:
    """
    Exports the final research results (from KnowledgeBase) to a file on the server.
    Requires RESEARCH_COMPLETE state.
    Input: ExportResultsArgs (specifying format: csv, json, parquet).
    Output: ExportResult containing the server-side file path.
    """
    operation = "handle_export_results"
    logger.info(f"Tool '{operation}' called with format: {args.format.value}")
    # Access components via the instances registry
    state_manager = instances.state_manager
    kb = instances.knowledge_base
    artifacts_dir = "artifacts"  # Relative to project root

    try:
        if not state_manager or not kb:
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details="Required server components (StateManager or KnowledgeBase) not initialized.",
            )
            raise AYAServerError(
                "Server configuration error: Core components missing.", data=op_data
            )

        # Allow export also in AWAITING_USER_CLARIFICATION state as per docs/roadmap
        allowed_states = [
            OverallStatus.RESEARCH_COMPLETE,
            OverallStatus.AWAITING_USER_CLARIFICATION,
        ]
        if state_manager.status not in allowed_states:
            op_data = OperationalErrorData(
                component="StateManager",
                operation=operation,
                details=f"Invalid state: {state_manager.status.value}",
            )
            raise AYAServerError(
                f"Cannot export results in current state: {state_manager.status.value}. Requires RESEARCH_COMPLETE or AWAITING_USER_CLARIFICATION.",
                code=INVALID_REQUEST,
                data=op_data,
            )

        # Ensure artifacts directory exists
        try:
            os.makedirs(artifacts_dir, exist_ok=True)
            logger.debug(f"Ensured artifacts directory exists: {artifacts_dir}")
        except (OSError, PermissionError) as dir_err:
            op_data = OperationalErrorData(
                component="Filesystem",
                operation="makedirs",
                details=f"Failed to create artifacts directory '{artifacts_dir}': {dir_err}",
            )
            raise AYAServerError(
                "Filesystem error: Could not create export directory.",
                data=op_data,
            ) from dir_err

        # Generate unique filename and path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        file_format = args.format.value
        filename = f"export_{timestamp}_{unique_id}.{file_format}"
        full_file_path = os.path.join(artifacts_dir, filename)
        logger.debug(f"Generated export file path: {full_file_path}")

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

        # Write DataFrame directly to file based on format
        try:
            # Create a copy for modification before export
            df_to_export = all_data_df.copy()

            if args.format == ExportFormat.CSV:
                # Serialize complex object columns to JSON strings for CSV export
                logger.debug(
                    f"Serializing internal columns to JSON for CSV export: {INTERNAL_COLS}"
                )
                for col in INTERNAL_COLS:
                    if col in df_to_export.columns:
                        # Check if the column actually contains list/dict before applying serialization
                        # This check avoids errors if a column expected to be complex isn't, or is all null
                        needs_serialization = (
                            df_to_export[col]
                            .apply(lambda x: isinstance(x, (list, dict)))
                            .any()
                        )
                        if needs_serialization:
                            logger.debug(f"Serializing column '{col}' to JSON.")
                            # Apply json.dumps only to non-null values that are lists or dicts
                            df_to_export[col] = df_to_export[col].apply(
                                lambda x: (
                                    json.dumps(x) if isinstance(x, (list, dict)) else x
                                )
                            )
                # Export the modified DataFrame
                df_to_export.to_csv(full_file_path, index=False)
            elif args.format == ExportFormat.JSON:
                all_data_df.to_json(
                    full_file_path, orient="records", lines=True, index=False
                )
            elif args.format == ExportFormat.PARQUET:
                all_data_df.to_parquet(full_file_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {args.format}")

            logger.info(
                f"Successfully exported {len(all_data_df)} rows to {full_file_path}"
            )
        except (OSError, PermissionError, ValueError) as write_err:
            op_data = OperationalErrorData(
                component="Filesystem",
                operation=f"to_{file_format}",
                details=f"Failed to write export file '{full_file_path}': {write_err}",
            )
            raise AYAServerError(
                "Filesystem error: Could not write export file.", data=op_data
            ) from write_err
        except Exception as unexpected_write_err:
            op_data = OperationalErrorData(
                component="PandasExport",
                operation=f"to_{file_format}",
                details=f"Unexpected error writing export file '{full_file_path}': {unexpected_write_err}",
            )
            raise AYAServerError(
                "Unexpected error during file export.", data=op_data
            ) from unexpected_write_err

        # Return the result object with the file path
        return ExportResult(file_path=full_file_path)

    except (AYAServerError, PydanticValidationError, Exception) as e:
        # Raise the converted McpError, preserving the original exception context
        raise handle_aya_exception(e, operation) from e
