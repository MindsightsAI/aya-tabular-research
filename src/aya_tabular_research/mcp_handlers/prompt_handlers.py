import logging
from typing import List

from mcp import types as mcp_types

# Import models needed for type checking and accessing data
from ..core.models.research_models import (
    InstructionObjectV3,
    StrategicReviewDirective,
)

# Import global component instances
from ..mcp_interface import _state_manager_instance

logger = logging.getLogger(__name__)


async def handle_overview() -> List[mcp_types.PromptMessage]:
    """Provides a high-level overview prompt based on the current state."""
    operation = "handle_overview"
    logger.debug(f"Prompt '{operation}' accessed.")
    state_manager = _state_manager_instance
    try:
        if not state_manager:
            logger.warning(f"{operation}: StateManager not initialized.")
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
            active_instr_focus = (
                f"Strategic Review ({state_manager.active_instruction.review_reason})"
            )

        overview = f"""
System Overview:
- Current State: {status.value}
- Task Goal: {task_desc}
- Active Focus: {active_instr_focus}
- Available Tools: {', '.join(state_manager.get_tools_for_state(status))}

Provide guidance or use an available tool.
"""
        return [mcp_types.PromptMessage(role="system", content=overview.strip())]
    except Exception as e:
        logger.error(f"Error generating overview prompt: {e}", exc_info=True)
        # Return a simple error message within the prompt structure
        return [
            mcp_types.PromptMessage(
                role="system", content=f"Error generating overview: {e}"
            )
        ]
