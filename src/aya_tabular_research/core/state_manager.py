import logging
from datetime import datetime, timezone
from typing import (  # Awaitable, # Removed; Callable, # Removed
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Union,
)

# Import KnowledgeBase directly for runtime checks
from ..storage.knowledge_base import KnowledgeBase

# Import custom exceptions and error models
from .exceptions import AYAServerError, KBInteractionError
from .models.enums import InquiryStatus, OverallStatus
from .models.error_models import OperationalErrorData
from .models.research_models import (  # Added StrategicReviewDirective
    InstructionObjectV3,
    StrategicReviewDirective,
    TaskDefinition,
)

if TYPE_CHECKING:
    # Keep KB import here for type checkers if needed, but it's also available from above
    pass  # Or remove the TYPE_CHECKING block if KB is the only thing in it

logger = logging.getLogger(__name__)

# NotifierCallback removed


class StateManager:
    """
    Manages the overall state of the research task for a single server instance.
    Handles status transitions, stores task definition and active instruction.
    Triggers state change notifications via MCP. This version is STATELESS
    and does not persist state between server runs.
    """

    def __init__(self, knowledge_base: "KnowledgeBase"):  # Removed mcp_notifier
        # Use the imported KnowledgeBase for the runtime check
        if not isinstance(
            knowledge_base, KnowledgeBase
        ):  # Runtime check uses the imported class
            raise TypeError("knowledge_base must be an instance of KnowledgeBase")
        # Removed mcp_notifier check
        self.knowledge_base = knowledge_base
        # self._notify = None # Removed notifier storage
        # Always start in the initial state
        self._status: OverallStatus = OverallStatus.AWAITING_TASK_DEFINITION
        # Added STRATEGIC_REVIEW to the Literal
        self.last_processed_directive_type: Optional[
            Literal["DISCOVERY", "ENRICHMENT", "STRATEGIC_REVIEW"]
        ] = None
        self._task_definition: Optional[TaskDefinition] = None
        # Updated type hint to Union
        self._active_instruction: Optional[
            Union[InstructionObjectV3, StrategicReviewDirective]
        ] = None
        self._active_instruction_start_time: Optional[datetime] = None
        self._clarification_needed_reason: Optional[str] = None
        logger.info(
            f"StateManager initialized. Initial status: {self._status.value} (Stateless Mode)"
        )

    # --- Properties ---
    @property
    def status(self) -> OverallStatus:
        return self._status

    @property
    def task_definition(self) -> Optional[TaskDefinition]:
        return self._task_definition

    @property
    # Updated return type hint
    def active_instruction(
        self,
    ) -> Optional[Union[InstructionObjectV3, StrategicReviewDirective]]:
        return self._active_instruction

    @property
    def active_instruction_start_time(self) -> Optional[datetime]:
        """Get the start time of the currently active instruction (if any)."""
        return self._active_instruction_start_time

    @property
    def clarification_needed_reason(self) -> Optional[str]:
        return self._clarification_needed_reason

    # --- Private Helper Methods ---

    def _clear_active_instruction(self):
        """Internal helper to clear instruction and its start time."""
        if self._active_instruction:
            # Handle different ID attribute names
            active_id = getattr(
                self._active_instruction,
                "instruction_id",
                getattr(self._active_instruction, "directive_id", "N/A"),
            )
            logger.debug(f"Clearing active instruction {active_id} and start time.")
        self._active_instruction = None
        self._active_instruction_start_time = None

    def get_tools_for_state(self, status: OverallStatus) -> List[str]:
        """Returns the list of tool names available for a given status."""
        mapping = {
            OverallStatus.AWAITING_TASK_DEFINITION: ["research/define_task"],
            OverallStatus.AWAITING_DIRECTIVE: [
                "research/preview_results",
                "research/export_results",
            ],
            OverallStatus.CONDUCTING_INQUIRY: ["research/submit_inquiry_report"],
            OverallStatus.AWAITING_USER_CLARIFICATION: [
                "research/submit_user_clarification",
                "research/preview_results",
                "research/export_results",
            ],
            OverallStatus.RESEARCH_COMPLETE: [
                "research/preview_results",
                "research/export_results",
            ],
            OverallStatus.FATAL_ERROR: [],
        }
        return mapping.get(status, [])

    async def _transition_state(self, new_status: OverallStatus):
        """Handles state transition and notification."""
        if self._status == new_status:
            logger.debug(
                f"Attempted transition to the same state: {new_status.value}. No action taken."
            )
            return  # No change

        old_status = self._status
        self._status = new_status
        logger.info(
            f"Transitioning state from {old_status.value} -> {new_status.value}"
        )

        # Clear specific fields based on new state
        if new_status != OverallStatus.CONDUCTING_INQUIRY:
            self._clear_active_instruction()
        if new_status != OverallStatus.AWAITING_USER_CLARIFICATION:
            self._clarification_needed_reason = None

        # Determine available tools
        # Prepare and send notification
        # Removed notification sending logic
        logger.debug(
            f"State transitioned to {new_status.value}. Notification sending removed."
        )

    # --- Core State Logic ---

    # Updated parameter type hint
    def set_active_instruction(
        self,
        instruction: Optional[Union[InstructionObjectV3, StrategicReviewDirective]],
    ):
        """Sets the active instruction and records its start time."""
        self._active_instruction = instruction
        if instruction:
            self._active_instruction_start_time = datetime.now(timezone.utc)
            # Handle different ID attribute names
            active_id = getattr(
                instruction,
                "instruction_id",
                getattr(instruction, "directive_id", "N/A"),
            )
            logger.debug(
                f"Active instruction set: {active_id} at {self._active_instruction_start_time.isoformat()}"
            )
        else:
            self._active_instruction_start_time = None
            logger.debug("Active instruction explicitly cleared.")

    async def define_task(self, task_def: TaskDefinition):
        """
        Validates task definition, initializes KB, and transitions state.
        Raises TaskValidationError or KBInteractionError on failure.
        """
        if self._status != OverallStatus.AWAITING_TASK_DEFINITION:
            # This is an internal logic error, should ideally not happen if interface layer checks state
            logger.error(f"Cannot define task in status: {self._status.value}")
            # Raise an operational error rather than returning False
            op_data = OperationalErrorData(
                component="StateManager",
                operation="define_task",
                details=f"Invalid state: {self._status.value}",
            )
            raise AYAServerError(
                f"Cannot define task in current state: {self._status.value}",
                data=op_data,
            )  # Or a more specific StateTransitionError if defined

        # TaskDefinition validation happens implicitly during Pydantic parsing
        # in the MCPInterface layer. If we reach here, task_def is valid.
        self._task_definition = task_def
        logger.info("Task definition stored in StateManager.")

        try:
            # Initialize KB (will be in-memory for this instance)
            self.knowledge_base.initialize(task_def)
            if task_def.seed_entities:
                # Use batch add for efficiency even during seeding
                self.knowledge_base.batch_update_entities(task_def.seed_entities)

            self.last_processed_directive_type = None  # Reset on new task
            await self._transition_state(OverallStatus.AWAITING_DIRECTIVE)
            # Removed return True, success is indicated by not raising an exception

        except Exception as e:
            # Catch specific KB errors if KB layer raises them, otherwise catch generic Exception
            logger.error(
                f"Failed during KB initialization or seeding: {e}", exc_info=True
            )
            # Raise specific KBInteractionError
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation="initialize/seed",
                details=str(e),
                is_retryable=False,  # Usually KB init failure isn't retryable without changes
            )
            raise KBInteractionError(
                f"Task Definition/KB Init failed: {e}", operation_data=op_data
            ) from e
            # Removed set_fatal_error and return False

    # Updated parameter type hint
    async def directive_issued(
        self, instruction: Union[InstructionObjectV3, StrategicReviewDirective]
    ):
        """Sets the active instruction and transitions state to CONDUCTING_INQUIRY."""
        if self._status != OverallStatus.AWAITING_DIRECTIVE:
            logger.error(f"Cannot issue directive in status: {self._status.value}")
            return

        self.set_active_instruction(instruction)
        await self._transition_state(OverallStatus.CONDUCTING_INQUIRY)

    async def report_received(
        self,
        inquiry_status: InquiryStatus,
        # Added STRATEGIC_REVIEW to the Literal
        directive_type: Literal["DISCOVERY", "ENRICHMENT", "STRATEGIC_REVIEW"],
        obstacle_summary: Optional[str] = None,
    ):
        """Processes report outcome, transitions state based on client status."""
        if self._status != OverallStatus.CONDUCTING_INQUIRY:
            logger.error(
                f"Received report outcome, but not in CONDUCTING_INQUIRY state (current: {self._status.value}). Ignoring."
            )
            # Store the type of the directive that was just processed
            self.last_processed_directive_type = directive_type
            logger.debug(
                f"Stored last processed directive type: {self.last_processed_directive_type}"
            )
            return

        # Handle different ID attribute names
        active_id = (
            getattr(
                self._active_instruction,
                "instruction_id",
                getattr(self._active_instruction, "directive_id", "N/A"),
            )
            if self._active_instruction
            else "N/A"
        )
        logger.info(
            f"Report outcome received for instruction {active_id}. Client status: {inquiry_status.value}"
        )

        next_status = OverallStatus.AWAITING_DIRECTIVE  # Default
        if inquiry_status == InquiryStatus.BLOCKED:
            next_status = OverallStatus.AWAITING_USER_CLARIFICATION
            self._clarification_needed_reason = (
                obstacle_summary or "Client reported BLOCKED status."
            )
            logger.warning(
                f"Inquiry blocked. Reason: {self._clarification_needed_reason}. Transitioning to {next_status.value}"
            )
        elif inquiry_status not in [
            InquiryStatus.COMPLETED,
            InquiryStatus.PARTIAL,
            InquiryStatus.FAILED,
        ]:
            logger.error(
                f"Received unexpected InquiryStatus: {inquiry_status}. Defaulting to AWAITING_DIRECTIVE."
            )

        logger.info(
            f"Report processed. Determined next state: {next_status.value}. Last directive type was: {self.last_processed_directive_type}"
        )
        await self._transition_state(next_status)

    async def clarification_received(self, clarification_data: Optional[Any] = None):
        """Processes clarification and transitions state back to AWAITING_DIRECTIVE."""
        if self._status != OverallStatus.AWAITING_USER_CLARIFICATION:
            logger.error(
                f"Received clarification, but not in AWAITING_USER_CLARIFICATION state (current: {self._status.value}). Ignoring."
            )
            return

        logger.info(f"User clarification received. Data: {clarification_data}")
        # TODO: Potentially store clarification_data in KB or pass to Planner

        await self._transition_state(OverallStatus.AWAITING_DIRECTIVE)

    async def request_clarification(self, reason: str):
        """Transitions state to AWAITING_USER_CLARIFICATION."""
        if self._status not in [
            OverallStatus.AWAITING_DIRECTIVE,
            OverallStatus.CONDUCTING_INQUIRY,
        ]:
            logger.error(
                f"Cannot request clarification in status: {self._status.value}."
            )
            return

        logger.warning(f"Requesting user clarification. Reason: {reason}")
        self._clarification_needed_reason = reason
        await self._transition_state(OverallStatus.AWAITING_USER_CLARIFICATION)

    async def complete_research(self):
        """Sets the status to RESEARCH_COMPLETE."""
        if self._status == OverallStatus.RESEARCH_COMPLETE:
            return
        await self._transition_state(OverallStatus.RESEARCH_COMPLETE)

    async def set_fatal_error(self, message: str):
        """Sets the status to FATAL_ERROR."""
        if self._status == OverallStatus.FATAL_ERROR:
            return
        logger.critical(f"Fatal error occurred: {message}.")
        await self._transition_state(OverallStatus.FATAL_ERROR)

    async def reset_state(self):
        """Resets the state manager to its initial condition (in-memory)."""
        logger.warning("Resetting StateManager (in-memory).")
        self._task_definition = None
        # Transition to initial state (clears other fields, notifies)
        self.last_processed_directive_type = None  # Reset on state reset
        await self._transition_state(OverallStatus.AWAITING_TASK_DEFINITION)
