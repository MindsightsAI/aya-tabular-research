import datetime
import io  # For CSV export
import logging
import os
import uuid
from typing import TYPE_CHECKING, Optional, Union

from mcp import types as mcp_types
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_REQUEST, ErrorData

# Alias Pydantic's ValidationError
# Import BaseModel for type checking in helper
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

# Import custom exceptions and error models
from .core.exceptions import (  # KBInteractionError, # Caught by AYAServerError; PlanningError, # Caught by AYAServerError; ReportProcessingError, # Caught by AYAServerError; TaskValidationError, # Caught by AYAServerError or PydanticValidationError
    AYAServerError,
)
from .core.models import research_models
from .core.models.enums import OverallStatus
from .core.models.error_models import (
    FieldErrorDetail,
    OperationalErrorData,
    ValidationErrorData,
)

# Import the Union type for combined results payload
from .core.models.research_models import ExportFormat  # Added
from .core.models.research_models import ExportResult  # Added
from .core.models.research_models import ExportResultsArgs  # Added
from .core.models.research_models import (
    DefineTaskArgs,
    DefineTaskResult,
    InstructionObjectV3,
    ServerStatusPayload,
    StrategicReviewDirective,
    SubmitClarificationResult,
    SubmitInquiryReportArgs,
    SubmitUserClarificationArgs,
)
from .core.state_manager import StateManager
from .execution.instruction_builder import DirectiveBuilder
from .execution.report_handler import ReportHandler
from .planning.planner import Planner
from .storage.knowledge_base import KnowledgeBase

# Use TYPE_CHECKING to import AppContext only for type hinting, avoiding circular import at runtime
if TYPE_CHECKING:
    pass  # No specific type hints needed here currently

logger = logging.getLogger(__name__)

# --- Global placeholders for components ---
_state_manager_instance: Optional[StateManager] = None
_planner_instance: Optional[Planner] = None
_directive_builder_instance: Optional[DirectiveBuilder] = None
_report_handler_instance: Optional[ReportHandler] = None
_knowledge_base_instance: Optional[KnowledgeBase] = None


def register_mcp_handlers(mcp_server: FastMCP):
    """Registers MCP tool and prompt handlers with the FastMCP server instance."""

    # --- Helper to convert custom errors to McpError ---
    def _handle_aya_exception(e: Exception, operation: str) -> McpError:
        """Converts AYAServerError, PydanticValidationError, or generic Exception into McpError."""
        if isinstance(e, AYAServerError):
            logger.error(
                f"AYA Server Error during {operation}: {e.message}", exc_info=False
            )
            data_payload = (
                e.data.model_dump(mode="json")
                if isinstance(e.data, BaseModel)
                else e.data
            )
            # NOTE: We return the McpError here, the caller should raise it with 'from e'
            return McpError(
                ErrorData(code=e.code, message=e.message, data=data_payload)
            )
        elif isinstance(e, PydanticValidationError):
            logger.warning(
                f"Pydantic Validation Error during {operation}: {e}", exc_info=False
            )
            details = [
                FieldErrorDetail(
                    field=".".join(map(str, err.get("loc", ["unknown"]))),
                    error=err.get("msg", "Unknown validation error"),
                )
                for err in e.errors()
            ]
            validation_data = ValidationErrorData(
                detail=details,
                suggestion="Check input arguments against the tool/model schema.",
            )
            # NOTE: We return the McpError here, the caller should raise it with 'from e'
            return McpError(
                ErrorData(
                    code=INVALID_REQUEST,
                    message="Input validation failed.",
                    data=validation_data.model_dump(mode="json"),
                )
            )
        else:
            logger.critical(
                f"Unhandled exception during {operation}: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="MCPInterface",
                operation=operation,
                details=f"Unexpected error: {type(e).__name__}",
            )
            # NOTE: We return the McpError here, the caller should raise it with 'from e'
            return McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="An unexpected internal server error occurred.",
                    data=op_data.model_dump(mode="json"),
                )
            )

    # --- Tool Handlers ---

    @mcp_server.tool("research/define_task")
    async def handle_define_task(
        args: DefineTaskArgs, ctx: Context
    ) -> DefineTaskResult:
        """
        Defines the research scope and parameters. Requires AWAITING_TASK_DEFINITION state.
        Input: DefineTaskArgs (containing TaskDefinition).
        Output: DefineTaskResult, which includes confirmation and the *first* directive object
                (InstructionObjectV3 or StrategicReviewDirective with embedded context)
                within the 'instruction_payload' field, or a status update.
        """
        operation = "handle_define_task"
        logger.info(f"Tool '{operation}' called.")
        state_manager = _state_manager_instance
        planner = _planner_instance
        directive_builder = _directive_builder_instance

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
                    directive_type="ENRICHMENT",
                )
                result_status = (
                    research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                )
            elif planner_signal_tuple is None:
                await state_manager.complete_research()
                next_directive = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus="Research Complete",
                    allowed_tools=[
                        "research/preview_results",
                        "research/export_results",
                    ],
                    directive_type="ENRICHMENT",
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
            raise _handle_aya_exception(e, operation) from e

    @mcp_server.tool("research/submit_inquiry_report")
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
        state_manager = _state_manager_instance
        report_handler = _report_handler_instance
        planner = _planner_instance
        directive_builder = _directive_builder_instance

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
                logger.info(
                    f"Acting on client strategic decision: {strategic_decision}"
                )
                if strategic_decision == "FINALIZE":
                    await state_manager.complete_research()
                    result_status = (
                        research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
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
                    else:
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
                    else:
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
                    client_assessment=None
                )
                if planner_signal_tuple == "CLARIFICATION_NEEDED":
                    planner_signal = ("NEEDS_STRATEGIC_REVIEW", "critical_obstacles")
                elif planner_signal_tuple is None:
                    await state_manager.complete_research()
                    result_status = (
                        research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
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
                else:
                    op_data = OperationalErrorData(
                        component="StateManager",
                        operation="report_received",
                        details=f"Unexpected state {resulting_status.value} after report processing.",
                    )
                    raise AYAServerError(
                        f"Unexpected state after report: {resulting_status.value}",
                        data=op_data,
                    )

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
                summary_text = "User clarification needed."
            elif next_directive_payload:
                summary_text = f"Next step: {getattr(next_directive_payload, 'inquiry_focus', 'Unknown')}"

            return research_models.SubmitReportAndGetDirectiveResult(
                message="Report submitted successfully.",
                status=result_status,
                summary=summary_text,
                instruction_payload=next_directive_payload,
                error_message=None,
            )
        except (AYAServerError, PydanticValidationError, Exception) as e:
            raise _handle_aya_exception(e, operation) from e

    @mcp_server.tool("research/submit_user_clarification")
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
        state_manager = _state_manager_instance
        planner = _planner_instance
        directive_builder = _directive_builder_instance

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
            planner.incorporate_clarification(args.clarification)
            await state_manager.clarification_received(args.clarification)
            logger.info(
                f"Clarification received and processed. New status: {state_manager.status.value}"
            )

            # Plan next directive after clarification
            logger.debug("Planning next directive after clarification.")
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

            planner_signal_tuple = planner.determine_next_directive()

            if planner_signal_tuple == "CLARIFICATION_NEEDED":
                # If planner *still* needs clarification, something is wrong or the user input wasn't sufficient
                # For now, re-request clarification with more context
                clarification_reason = "Further clarification needed after user input."
                await state_manager.request_clarification(clarification_reason)
                result_status = (
                    research_models.RequestDirectiveResultStatus.CLARIFICATION_NEEDED
                )
                # Optionally build a specific directive asking for more info
                next_directive_payload = InstructionObjectV3(
                    research_goal_context=task_def.task_description,
                    inquiry_focus=f"Further Clarification Needed: {clarification_reason}",
                    focus_areas=[
                        "Please provide more specific guidance via 'research/submit_user_clarification'."
                    ],
                    allowed_tools=["research/submit_user_clarification"],
                    directive_type="ENRICHMENT",
                )

            elif planner_signal_tuple is None:
                await state_manager.complete_research()
                result_status = (
                    research_models.RequestDirectiveResultStatus.RESEARCH_COMPLETE
                )
            else:
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
            current_status_str = state_manager.status.value
            available_tools = state_manager.get_available_tools()

            return SubmitClarificationResult(
                message="Clarification submitted successfully.",
                status=result_status,
                summary=summary_text,
                instruction_payload=next_directive_payload,
                error_message=None,
                new_status=current_status_str,
                available_tools=available_tools,
            )
        except (AYAServerError, PydanticValidationError, Exception) as e:
            raise _handle_aya_exception(e, operation) from e

    # --- Resource Handlers ---

    @mcp_server.resource("research://research/status")
    async def get_server_status() -> ServerStatusPayload:
        """Provides the current status of the research task and available tools."""
        operation = "get_server_status"
        logger.debug(f"Resource '{operation}' accessed.")
        state_manager = _state_manager_instance
        try:
            if not state_manager:
                op_data = OperationalErrorData(
                    component="MCPInterface",
                    operation=operation,
                    details="StateManager not initialized.",
                )
                raise AYAServerError(
                    "Server configuration error: StateManager missing.", data=op_data
                )

            status_str = state_manager.status.value
            tools = state_manager.get_available_tools()
            task_defined = state_manager.task_definition is not None
            active_instr_id = (
                state_manager.active_instruction.instruction_id
                if isinstance(state_manager.active_instruction, InstructionObjectV3)
                else (
                    state_manager.active_instruction.directive_id
                    if isinstance(
                        state_manager.active_instruction, StrategicReviewDirective
                    )
                    else None
                )
            )

            return ServerStatusPayload(
                status=status_str,
                available_tools=tools,
                task_defined=task_defined,
                active_instruction_id=active_instr_id,
            )
        except (AYAServerError, Exception) as e:
            raise _handle_aya_exception(e, operation) from e

    @mcp_server.resource("debug://research/debug_state")
    async def get_debug_state() -> str:
        """FOR DEBUGGING ONLY: Returns a JSON string representation of the current state."""
        operation = "get_debug_state"
        logger.debug(f"Resource '{operation}' accessed.")
        state_manager = _state_manager_instance
        kb = _knowledge_base_instance
        try:
            if not state_manager or not kb:
                return '{"error": "Components not initialized"}'

            # Use Pydantic for serialization where possible
            class TempDebugModel(BaseModel):
                state_manager_status: str
                task_definition: Optional[research_models.TaskDefinition]
                active_instruction: Optional[
                    Union[InstructionObjectV3, StrategicReviewDirective]
                ]
                kb_summary: dict

            debug_data = TempDebugModel(
                state_manager_status=state_manager.status.value,
                task_definition=state_manager.task_definition,
                active_instruction=state_manager.active_instruction,
                kb_summary=kb.get_knowledge_summary(),
            )
            return debug_data.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Error generating debug state: {e}", exc_info=True)
            return f'{{"error": "Failed to generate debug state: {e}"}}'

    # --- Tools for Research Completion Phase ---

    @mcp_server.tool("research/preview_results")
    async def handle_preview_results(args: dict, ctx: Context) -> str:
        """
        Provides a preview (e.g., first 10 rows) of the final results as a CSV string.
        Requires RESEARCH_COMPLETE state.
        Input: Empty dictionary (args ignored).
        Output: String containing the CSV preview data.
        """
        operation = "handle_preview_results"
        logger.info(f"Tool '{operation}' called.")
        state_manager = _state_manager_instance
        kb = _knowledge_base_instance

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

            if state_manager.status != OverallStatus.RESEARCH_COMPLETE:
                op_data = OperationalErrorData(
                    component="StateManager",
                    operation=operation,
                    details=f"Invalid state: {state_manager.status.value}",
                )
                raise AYAServerError(
                    f"Cannot preview results in current state: {state_manager.status.value}",
                    code=INVALID_REQUEST,
                    data=op_data,
                )

            # Fetch preview data
            preview_df = kb.get_data_preview(num_rows=10)

            if preview_df.empty:
                logger.warning("Preview requested, but Knowledge Base is empty.")
                return "No data available for preview."

            # Convert preview DataFrame to CSV string
            csv_buffer = io.StringIO()
            preview_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            csv_buffer.close()

            logger.info(
                f"Successfully generated preview CSV string (length: {len(csv_string)})."
            )
            return csv_string

        except (AYAServerError, PydanticValidationError, Exception) as e:
            raise _handle_aya_exception(e, operation) from e

    @mcp_server.tool("research/export_results")
    async def handle_export_results(
        args: ExportResultsArgs, ctx: Context
    ) -> ExportResult:
        """
        Exports the final research results (from KnowledgeBase) to a file on the server.
        Requires RESEARCH_COMPLETE state.
        Input: ExportResultsArgs (specifying format: csv, json, parquet).
        Output: ExportResult containing the server-side file path.
        """
        operation = "handle_export_results"
        logger.info(f"Tool '{operation}' called with format: {args.format.value}")
        state_manager = _state_manager_instance
        kb = _knowledge_base_instance
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

            if state_manager.status != OverallStatus.RESEARCH_COMPLETE:
                op_data = OperationalErrorData(
                    component="StateManager",
                    operation=operation,
                    details=f"Invalid state: {state_manager.status.value}",
                )
                raise AYAServerError(
                    f"Cannot export results in current state: {state_manager.status.value}. Requires RESEARCH_COMPLETE.",
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
            try:
                all_data_df = kb.get_all_data_as_dataframe()
                logger.debug(f"Fetched {len(all_data_df)} rows from KnowledgeBase.")
            except AYAServerError as kb_err:
                # Re-raise KB errors specifically if needed, otherwise caught below
                logger.error(
                    f"Failed to fetch data from KB for export: {kb_err.message}"
                )
                raise  # Let the generic handler catch and format it
            except Exception as fetch_err:
                # Catch unexpected errors during fetch
                op_data = OperationalErrorData(
                    component="KnowledgeBase",
                    operation="get_all_data_as_dataframe",
                    details=f"Unexpected error fetching data: {fetch_err}",
                )
                raise AYAServerError(
                    "Failed to retrieve data from Knowledge Base for export.",
                    data=op_data,
                ) from fetch_err

            if all_data_df.empty:
                logger.warning("Export requested, but Knowledge Base is empty.")
                # Still create an empty file as requested
                # Fall through to writing logic which will create an empty file

            # Write DataFrame directly to file based on format
            try:
                if args.format == ExportFormat.CSV:
                    all_data_df.to_csv(full_file_path, index=False)
                elif args.format == ExportFormat.JSON:
                    # Use line-delimited JSON for better streaming/large file handling
                    all_data_df.to_json(
                        full_file_path, orient="records", lines=True, index=False
                    )
                elif args.format == ExportFormat.PARQUET:
                    all_data_df.to_parquet(full_file_path, index=False)
                else:
                    # Should be caught by Pydantic enum validation, but belt-and-suspenders
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
            raise _handle_aya_exception(e, operation) from e

    # --- Prompt Handlers ---

    @mcp_server.prompt("research/overview")
    async def handle_overview() -> list[mcp_types.PromptMessage]:
        """Provides a high-level overview prompt based on the current state."""
        operation = "handle_overview"
        logger.debug(f"Prompt '{operation}' accessed.")
        state_manager = _state_manager_instance
        try:
            if not state_manager:
                return [
                    mcp_types.PromptMessage(
                        role="system", content="Error: StateManager not initialized."
                    )
                ]

            status = state_manager.status
            task_desc = (
                state_manager.task_definition.task_description
                if state_manager.task_definition
                else "Not defined"
            )
            active_instr_focus = "None"
            if isinstance(state_manager.active_instruction, InstructionObjectV3):
                active_instr_focus = state_manager.active_instruction.inquiry_focus
            elif isinstance(state_manager.active_instruction, StrategicReviewDirective):
                active_instr_focus = f"Strategic Review ({state_manager.active_instruction.review_reason})"

            overview = f"""
System Overview:
- Current State: {status.value}
- Task Goal: {task_desc}
- Active Focus: {active_instr_focus}
- Available Tools: {', '.join(state_manager.get_available_tools())}

Provide guidance or use an available tool.
"""
            return [mcp_types.PromptMessage(role="system", content=overview.strip())]
        except Exception as e:
            logger.error(f"Error generating overview prompt: {e}", exc_info=True)
            return [
                mcp_types.PromptMessage(
                    role="system", content=f"Error generating overview: {e}"
                )
            ]


# --- Global Component Setter (Called during server lifespan) ---


def set_global_components(
    state_manager: StateManager,
    planner: Planner,
    directive_builder: DirectiveBuilder,
    report_handler: ReportHandler,
    knowledge_base: KnowledgeBase,
):
    """Sets the global component instances used by the handlers."""
    global _state_manager_instance, _planner_instance, _directive_builder_instance, _report_handler_instance, _knowledge_base_instance
    _state_manager_instance = state_manager
    _planner_instance = planner
    _directive_builder_instance = directive_builder
    _report_handler_instance = report_handler
    _knowledge_base_instance = knowledge_base
    logger.debug("Global component instances set for MCP interface handlers.")
