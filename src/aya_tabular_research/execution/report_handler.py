import itertools  # Added for product generation
import logging
from typing import (  # Added Literal, Union; Added Tuple, Union
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import jsonschema
import pandas as pd

# Import MCP error types and helpers
from mcp.types import INVALID_REQUEST
from pydantic import BaseModel, Field

# Import custom exceptions and error models
from ..core.exceptions import (
    AYAServerError,
    KBInteractionError,
    ReportProcessingError,
)

# Import core application models and components
from ..core.models.enums import OverallStatus  # Added for return type
from ..core.models.enums import InquiryStatus
from ..core.models.error_models import (
    FieldErrorDetail,
    OperationalErrorData,
    ValidationErrorData,
)
from ..core.models.research_models import (
    StrategicReviewDirective,  # Import new directive type
)
from ..core.models.research_models import (  # SynthesizedFinding, IdentifiedObstacle, ProposedNextStep # Keep commented if not directly used
    InquiryReport,
    InstructionObjectV3,
    TaskDefinition,
)
from ..core.state_manager import StateManager
from ..storage.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class SchemaValidationErrorDetails(BaseModel):
    """Structured details for a schema validation error in a report data point."""

    error_type: str = "SchemaValidationError"
    message: str = Field(
        ..., description="The validation error message from jsonschema."
    )
    field_path: List[str | int] = Field(
        ..., description="Path to the field causing the error within the data point."
    )
    data_point_index: int = Field(
        ...,
        description="Index of the failing data point within structured_data_points.",
    )
    required_fields: Optional[List[str]] = Field(
        None, description="List of fields required by the schema, if applicable."
    )


class ReportHandler:
    """
    Processes incoming InquiryReports from the client synchronously for a stateless server instance.

    Responsibilities:
    1.  Validation of the report against the active instruction and schema.
    2.  Updating the Knowledge Base with validated data (using batch operations).
    3.  Signaling the outcome to the StateManager to trigger state transition.
    4.  Providing structured error feedback for validation failures.
    """

    def __init__(self, knowledge_base: KnowledgeBase, state_manager: StateManager):
        """Initializes the ReportHandler."""
        if not isinstance(knowledge_base, KnowledgeBase):
            raise TypeError("knowledge_base must be an instance of KnowledgeBase")
        if not isinstance(state_manager, StateManager):
            raise TypeError("state_manager must be an instance of StateManager")
        self._kb = knowledge_base
        self._state_manager = state_manager
        logger.info("ReportHandler initialized (Synchronous Mode).")

    def _validate_report(
        self, report: InquiryReport, active_instruction: InstructionObjectV3
    ) -> None:
        """
        Performs synchronous validation checks (ID match, schema).
        Raises ReportProcessingError on failure.
        """
        # 1. Instruction ID Match
        if report.instruction_id != active_instruction.instruction_id:
            msg = f"Report instruction ID '{report.instruction_id}' does not match active instruction '{active_instruction.instruction_id}'."
            logger.error(msg)
            op_data = OperationalErrorData(
                component="ReportHandler",
                operation="_validate_report",
                details=msg,
            )
            # Use INVALID_REQUEST code as it's likely a client error
            raise ReportProcessingError(msg, report_data=op_data, code=INVALID_REQUEST)

        # 2. Schema Validation
        guidelines = active_instruction.reporting_guidelines
        schema = None
        if guidelines and guidelines.required_schema_points:
            schema = {"type": "object", "required": guidelines.required_schema_points}
            logger.debug(
                f"Validating report against schema: {guidelines.required_schema_points}"
            )

            if not report.structured_data_points:
                logger.warning(
                    f"Schema requires points {guidelines.required_schema_points}, but structured_data_points is empty/None."
                )
                # This might be acceptable depending on the directive, not raising error here.
            else:
                for idx, data_point in enumerate(report.structured_data_points):
                    if not isinstance(data_point, dict):
                        msg = f"Data point at index {idx} is not a dictionary."
                        logger.error(f"Schema validation failed: {msg}")
                        field_error = FieldErrorDetail(
                            field=f"structured_data_points[{idx}]",
                            error="Must be a dictionary.",
                        )
                        validation_data = ValidationErrorData(detail=[field_error])
                        raise ReportProcessingError(
                            f"Report validation failed: Invalid data point type at index {idx}.",
                            report_data=validation_data,  # Use ValidationErrorData here
                            code=INVALID_REQUEST,
                        )
                    try:
                        jsonschema.validate(instance=data_point, schema=schema)
                    except jsonschema.ValidationError as e:
                        # Check if the error is specifically about missing required fields
                        if e.validator == "required":
                            logger.warning(
                                f"Partial data warning at index {idx}: Missing required field(s): {e.message}. Accepting partial data."
                            )
                            # Optionally store the warning or obstacle if needed later
                            # For now, just log and continue processing the partial data point
                            continue  # Skip raising error for missing required fields
                        else:
                            # For other schema errors (e.g., wrong type, format), raise the error
                            msg = f"Schema validation failed at index {idx} (validator: {e.validator}): {e.message}"
                            logger.error(msg)
                            field_path_str = (
                                ".".join(map(str, e.path)) if e.path else "root"
                            )
                            field_error = FieldErrorDetail(
                                field=field_path_str, error=e.message, value=e.instance
                            )
                            validation_data = ValidationErrorData(
                                detail=[field_error],
                                required_fields=schema.get("required"),
                            )
                            raise ReportProcessingError(
                                f"Report validation failed: {msg}",
                                report_data=validation_data,  # Use ValidationErrorData here
                                code=INVALID_REQUEST,
                            ) from e
                # If the loop completes without non-'required' errors for this data point
                logger.debug(
                    f"Schema validation passed (or only missing required fields) for data point at index {idx}."
                )
        # After checking all data points
        logger.debug("Schema validation checks completed for all data points.")

    def _update_knowledge_base(self, report: InquiryReport, task_def: TaskDefinition):
        """
        Updates the Knowledge Base synchronously using batch operations.
        Handles expansion of data points based on list values in granularity columns.
        Raises KBInteractionError on failure.
        """
        instruction_id = report.instruction_id
        logger.debug(
            f"Starting synchronous KB update for instruction: {instruction_id}"
        )
        try:
            identifier_col = task_def.identifier_column
            entities_to_update: List[Dict[str, Any]] = []
            primary_entity_id_for_feedback: Optional[str] = (
                None  # Track first ID for feedback
            )
            index_cols = (
                self._kb._data_store.index_columns
            )  # Get index columns from store
            malformed_count = 0  # Track malformed points during expansion

            # 1. Prepare structured data (potentially expanding list values)
            if report.structured_data_points:
                active_instruction = self._state_manager.active_instruction
                target_entity_id_from_instruction = (
                    active_instruction.target_entity_id if active_instruction else None
                )

                for idx, data_point in enumerate(report.structured_data_points):
                    if not isinstance(data_point, dict):
                        logger.warning(f"Skipping non-dict data point at index {idx}.")
                        malformed_count += 1
                        continue

                    # Determine the primary entity ID for this data point
                    entity_id_val = data_point.get(identifier_col)
                    current_entity_id: Optional[str] = None
                    if entity_id_val is not None and not pd.isnull(entity_id_val):
                        current_entity_id = str(entity_id_val)
                    elif target_entity_id_from_instruction:
                        current_entity_id = target_entity_id_from_instruction
                        logger.debug(
                            f"Using target entity ID '{current_entity_id}' for data point at index {idx}"
                        )
                    else:
                        logger.warning(
                            f"Skipping data point at index {idx} due to missing identifier ('{identifier_col}') and no target entity ID."
                        )
                        malformed_count += 1
                        continue

                    # Track the first entity ID encountered for feedback storage
                    if primary_entity_id_for_feedback is None:
                        primary_entity_id_for_feedback = current_entity_id

                    # Separate scalar index values and list index values
                    # Include identifier_col in template even if it's also an index col
                    record_template = {
                        k: v
                        for k, v in data_point.items()
                        if not (k in index_cols and isinstance(v, list))
                    }
                    record_template[identifier_col] = (
                        current_entity_id  # Ensure primary ID is set
                    )

                    list_cols_data = {
                        k: v
                        for k, v in data_point.items()
                        if k in index_cols and isinstance(v, list)
                    }

                    if not list_cols_data:
                        # Simple case: no lists in granularity columns
                        # Ensure all index columns are present
                        if all(
                            col in record_template
                            and not pd.isnull(record_template[col])
                            for col in index_cols
                        ):
                            entities_to_update.append(record_template)
                        else:
                            missing_req_idx = [
                                col
                                for col in index_cols
                                if col not in record_template
                                or pd.isnull(record_template.get(col))
                            ]
                            logger.warning(
                                f"Skipping data point at index {idx} for entity '{current_entity_id}' due to missing/null required index columns: {missing_req_idx}"
                            )
                            malformed_count += 1
                    else:
                        # Expansion case: generate product of list values
                        list_col_names = list(list_cols_data.keys())
                        list_col_values = list(list_cols_data.values())

                        # Check for empty lists which would lead to no combinations
                        if any(not v for v in list_col_values):
                            # Find which specific list(s) were empty
                            empty_cols = [
                                name
                                for name, val_list in zip(
                                    list_col_names, list_col_values, strict=True
                                )
                                if not val_list
                            ]
                            logger.warning(
                                f"Skipping data point at index {idx} for entity '{current_entity_id}' due to empty list in index column(s): {empty_cols}"
                            )
                            malformed_count += 1
                            continue

                        # --- Pre-expansion Type Validation ---
                        skip_datapoint_due_type = False
                        try:
                            # Check if the underlying datastore and its index are ready for dtype checks
                            if (
                                self._kb._data_store._df is not None
                                and self._kb._data_store._df.index.nlevels > 0
                            ):
                                for col_name, values_list in list_cols_data.items():
                                    if col_name in self._kb._data_store._df.index.names:
                                        target_dtype = self._kb._data_store._df.index.get_level_values(
                                            col_name
                                        ).dtype
                                        # Basic check: if target is numeric, ensure list items are numeric (or convertible)
                                        # Allow None/NaN as they are checked later during combination iteration
                                        if pd.api.types.is_numeric_dtype(target_dtype):
                                            if not all(
                                                isinstance(v, (int, float))
                                                or pd.api.types.is_number(v)
                                                or pd.isnull(v)
                                                for v in values_list
                                            ):
                                                logger.warning(
                                                    f"Skipping data point at index {idx} for entity '{current_entity_id}': Non-numeric value found in list for numeric index column '{col_name}'."
                                                )
                                                skip_datapoint_due_type = True
                                                break
                                        # Add more specific checks if needed (e.g., for datetime, string length)
                                    else:
                                        # This case should ideally not happen if index_cols is accurate
                                        logger.warning(
                                            f"Index column '{col_name}' used for expansion not found in DataStore index names during type check."
                                        )

                        except (AttributeError, TypeError) as type_val_err:
                            # Handle cases where _data_store or _df might not be fully initialized yet for dtype checks,
                            # or attributes are not of the expected type.
                            logger.warning(
                                f"Could not perform pre-expansion type validation due to state/type issue: {type_val_err}. DataStore state insufficient or invalid."
                            )
                        except Exception as e:
                            logger.error(
                                f"Error during pre-expansion type validation for data point {idx}: {e}",
                                exc_info=True,
                            )
                            # Decide if this should halt processing or just warn - warning for now
                            # skip_datapoint_due_type = True # Uncomment to skip on error

                        if skip_datapoint_due_type:
                            malformed_count += 1
                            continue
                        # --- End Pre-expansion Type Validation ---

                        for value_combination in itertools.product(*list_col_values):
                            new_record = record_template.copy()
                            valid_combination = True
                            for i, col_name in enumerate(list_col_names):
                                if pd.isnull(value_combination[i]):
                                    # Enhanced logging for null value skip
                                    logger.warning(
                                        f"Skipping expanded record for entity '{current_entity_id}' due to null value encountered for index column '{col_name}'. Combination: {value_combination}"
                                    )
                                    valid_combination = False
                                    malformed_count += 1
                                    break  # Skip this specific combination
                                new_record[col_name] = value_combination[i]

                            if not valid_combination:
                                continue

                            # Ensure all index columns are present after combination (should be guaranteed by logic above)
                            if all(col in new_record for col in index_cols):
                                entities_to_update.append(new_record)
                            else:  # Should not happen if logic is correct
                                missing_req_idx = [
                                    col for col in index_cols if col not in new_record
                                ]
                                logger.error(
                                    f"INTERNAL ERROR: Expanded record for entity '{current_entity_id}' missing required index columns: {missing_req_idx}. Combination: {value_combination}"
                                )
                                malformed_count += 1

            # 2. Perform Batch Update for Structured Data
            if entities_to_update:
                self._kb.batch_update_entities(
                    entities_to_update
                )  # This might raise KBInteractionError
                logger.info(
                    f"KB Update: Submitted {len(entities_to_update)} entities/rows for batch update (Instruction: {instruction_id})."
                )
            elif not malformed_count:  # Only log if no other warnings occurred
                logger.info(
                    f"KB Update: No valid structured data points to batch update (Instruction: {instruction_id})."
                )

            # 3. Store Richer Reporting Fields (Synchronously) - Applied to the first entity ID encountered
            if primary_entity_id_for_feedback:
                logger.debug(
                    f"KB Update: Storing richer report data against primary entity '{primary_entity_id_for_feedback}' (Instruction: {instruction_id})"
                )
                # These might raise KBInteractionError
                if report.summary_narrative is not None:
                    self._kb.store_narrative(
                        primary_entity_id_for_feedback, report.summary_narrative
                    )
                if report.synthesized_findings:
                    self._kb.store_findings(
                        primary_entity_id_for_feedback, report.synthesized_findings
                    )
                if report.identified_obstacles:
                    self._kb.store_obstacles(
                        primary_entity_id_for_feedback, report.identified_obstacles
                    )
                if report.proposed_next_steps:
                    self._kb.store_proposals(
                        primary_entity_id_for_feedback, report.proposed_next_steps
                    )
                if report.confidence_score is not None:
                    self._kb.store_confidence(
                        primary_entity_id_for_feedback, report.confidence_score
                    )
            elif (
                report.summary_narrative
                or report.synthesized_findings
                or report.identified_obstacles
                or report.proposed_next_steps
                or report.confidence_score is not None
            ):
                logger.warning(
                    f"KB Update: Report has richer data but no primary entity ID found (Instruction: {instruction_id}). Data not stored."
                )  # This might happen if all data points were malformed

            logger.info(
                f"Synchronous KB update finished for instruction: {instruction_id}"
            )

        except KBInteractionError:  # Catch specific KB error if raised by KB layer
            logger.error(
                f"KB interaction failed during update for instruction {instruction_id}",
                exc_info=True,
            )
            raise  # Re-raise the specific error
        except Exception as kb_exc:
            # Wrap generic exceptions in KBInteractionError
            logger.critical(
                f"Critical unexpected error during synchronous KB update (Instruction: {instruction_id}): {kb_exc}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation="_update_knowledge_base",
                details=str(kb_exc),
            )
            raise KBInteractionError(
                f"KB update failed for instruction {instruction_id}: {kb_exc}",
                operation_data=op_data,
            ) from kb_exc

    async def process_report(
        self,
        report: InquiryReport,
        active_directive: Union[InstructionObjectV3, StrategicReviewDirective],
    ) -> Tuple[Optional[OverallStatus], Optional[str], Optional[List[str]]]:
        """
        Handles submitted report synchronously: Validates, updates KB (if applicable),
        then signals StateManager.

        Args:
            report: The submitted InquiryReport.
            active_directive: The directive this report is responding to.

        Returns:
            A tuple containing:
                - Resulting OverallStatus after signaling StateManager (or None on signal failure).
                - The client's strategic_decision (if applicable, else None).
                - The client's strategic_targets (if applicable, else None).

        Raises:
            ReportProcessingError on validation failure.
            KBInteractionError on KB update failure (for non-strategic reports).
            AYAServerError on state signaling failure.
        """
        logger.info(
            f"PROCESS_REPORT: START processing report synchronously for instruction ID: {report.instruction_id}"
        )

        # --- Get Context ---
        task_def = self._state_manager.task_definition

        if not active_directive:
            # This indicates an internal logic error in the caller (MCPInterface)
            op_data = OperationalErrorData(
                component="ReportHandler",
                operation="process_report",
                details="active_directive argument was None.",
            )
            raise AYAServerError(
                "Internal error: process_report called without active_directive.",
                data=op_data,
            )
        if not task_def:
            # This indicates an internal logic error (state inconsistency)
            op_data = OperationalErrorData(
                component="ReportHandler",
                operation="process_report",
                details="TaskDefinition not set in StateManager.",
            )
            raise AYAServerError(
                "Internal error: TaskDefinition missing during report processing.",
                data=op_data,
            )

        # --- Differentiated Processing based on Directive Type ---
        strategic_decision: Optional[str] = None
        strategic_targets: Optional[List[str]] = None
        directive_type_for_state_manager: Literal[
            "DISCOVERY", "ENRICHMENT", "STRATEGIC_REVIEW"
        ]

        if isinstance(active_directive, StrategicReviewDirective):
            directive_type_for_state_manager = "STRATEGIC_REVIEW"
            logger.info(
                f"Processing report for STRATEGIC_REVIEW directive: {active_directive.directive_id}"
            )

            # --- Validation for Strategic Review Report ---
            strategic_decision = report.strategic_decision
            strategic_targets = report.strategic_targets

            if not strategic_decision:
                msg = "Report for STRATEGIC_REVIEW directive is missing the required 'strategic_decision' field."
                logger.error(msg)
                field_error = FieldErrorDetail(
                    field="strategic_decision",
                    error="Field is required for strategic review.",
                )
                validation_data = ValidationErrorData(detail=[field_error])
                raise ReportProcessingError(
                    msg, report_data=validation_data, code=INVALID_REQUEST
                )

            valid_decisions = {
                "FINALIZE",
                "DISCOVER",
                "ENRICH",
                "ENRICH_SPECIFIC",
                "CLARIFY_USER",
            }
            if strategic_decision not in valid_decisions:
                msg = f"Invalid 'strategic_decision' value: {strategic_decision}. Must be one of {valid_decisions}."
                logger.error(msg)
                field_error = FieldErrorDetail(
                    field="strategic_decision",
                    error=f"Invalid value. Must be one of {valid_decisions}",
                    value=strategic_decision,
                )
                validation_data = ValidationErrorData(detail=[field_error])
                raise ReportProcessingError(
                    msg, report_data=validation_data, code=INVALID_REQUEST
                )

            if strategic_decision == "ENRICH_SPECIFIC":
                if not strategic_targets:
                    msg = "Report provided 'ENRICH_SPECIFIC' decision but is missing the required 'strategic_targets' field."
                    logger.error(msg)
                    field_error = FieldErrorDetail(
                        field="strategic_targets",
                        error="Field is required when decision is ENRICH_SPECIFIC.",
                    )
                    validation_data = ValidationErrorData(detail=[field_error])
                    raise ReportProcessingError(
                        msg, report_data=validation_data, code=INVALID_REQUEST
                    )

                if not isinstance(strategic_targets, list) or not all(
                    isinstance(t, str) for t in strategic_targets
                ):
                    msg = "'strategic_targets' must be a list of strings."
                    logger.error(msg)
                    field_error = FieldErrorDetail(
                        field="strategic_targets",
                        error="Must be a list of strings.",
                        value=strategic_targets,
                    )
                    validation_data = ValidationErrorData(detail=[field_error])
                    raise ReportProcessingError(
                        msg, report_data=validation_data, code=INVALID_REQUEST
                    )

            logger.info(
                f"Strategic Review Report validation successful. Decision: {strategic_decision}, Targets: {strategic_targets}"
            )
            # Skip KB update for structured_data_points

        elif isinstance(active_directive, InstructionObjectV3):
            directive_type_for_state_manager = active_directive.directive_type
            logger.info(
                f"Processing report for {directive_type_for_state_manager} directive: {active_directive.instruction_id}"
            )

            # --- Standard Validation ---
            # _validate_report raises ReportProcessingError on failure
            self._validate_report(report, active_directive)
            logger.info(
                f"PROCESS_REPORT: Report validation successful for instruction: {report.instruction_id}"
            )

            # --- Standard Knowledge Base Update ---
            # _update_knowledge_base raises KBInteractionError on failure
            self._update_knowledge_base(report, task_def)
            logger.info(
                f"PROCESS_REPORT: KB update successful for instruction: {report.instruction_id}"
            )

        else:
            # Should not happen with Union type hint, but safety check
            msg = f"Received report for unexpected active directive type: {type(active_directive)}"
            logger.error(msg)
            op_data = OperationalErrorData(
                component="ReportHandler", operation="process_report", details=msg
            )
            raise AYAServerError(
                msg, data=op_data
            )  # Use base error for unexpected internal issues

        # --- Signal StateManager (after successful validation and KB update) ---
        logger.debug(
            f"PROCESS_REPORT: Entering StateManager signal block for instruction {report.instruction_id}"
        )
        obstacle_summary_for_state = None
        if report.status == InquiryStatus.BLOCKED and report.identified_obstacles:
            obstacle_summary_for_state = report.identified_obstacles[0].obstacle

        try:
            logger.debug(
                f"PROCESS_REPORT: Calling state_manager.report_received with status {report.status.value}"
            )
            await self._state_manager.report_received(
                inquiry_status=report.status,
                directive_type=directive_type_for_state_manager,
                obstacle_summary=obstacle_summary_for_state,
            )
            logger.info(
                f"PROCESS_REPORT: Successfully signaled report received to StateManager. Client status: {report.status.value}"
            )
            # Return the new status from the state manager
            return (
                self._state_manager.status,
                strategic_decision,
                strategic_targets,
            )
        except Exception as signal_exc:
            logger.critical(
                f"CRITICAL: Failed to signal report outcome to StateManager for instruction {report.instruction_id}: {signal_exc}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="StateManager",
                operation="report_received",
                details=str(signal_exc),
            )
            # This is a critical internal error
            raise AYAServerError(
                f"Failed to signal report outcome: {signal_exc}", data=op_data
            ) from signal_exc
