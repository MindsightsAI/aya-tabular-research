import logging
from typing import Optional, Union

from pydantic import BaseModel

# Import the central instances registry
from ..core import instances

# Import custom exceptions and error models
from ..core.exceptions import AYAServerError
from ..core.models import research_models
from ..core.models.error_models import OperationalErrorData
from ..core.models.research_models import (
    InstructionObjectV3,
    ServerStatusPayload,
    StrategicReviewDirective,
)

# Import the central error handler
from .error_handler import handle_aya_exception

logger = logging.getLogger(__name__)


async def get_server_status() -> ServerStatusPayload:
    """Provides the current status of the research task and available tools."""
    operation = "get_server_status"
    logger.debug(f"Resource '{operation}' accessed.")
    # Access state_manager via the instances registry
    state_manager = instances.state_manager
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
        current_status = state_manager.status  # Get the status enum object
        tools = state_manager.get_tools_for_state(current_status)
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
        # Let the MCP framework handle formatting this into an MCP error response
        # by raising the converted McpError
        raise handle_aya_exception(e, operation) from e


async def get_debug_state() -> str:
    """FOR DEBUGGING ONLY: Returns a JSON string representation of the current state."""
    operation = "get_debug_state"
    logger.debug(f"Resource '{operation}' accessed.")
    # Access components via the instances registry
    state_manager = instances.state_manager
    kb = instances.knowledge_base
    try:
        if not state_manager or not kb:
            logger.warning(f"{operation}: Components not initialized.")
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
        # For this debug endpoint, return a simple error JSON string directly
        logger.error(f"Error generating debug state: {e}", exc_info=True)
        return f'{{"error": "Failed to generate debug state: {e}"}}'
