import logging
from typing import List

from mcp import types as mcp_types

from ..core import instances

# Import models needed for type checking and accessing data
from ..core.models.research_models import (
    InstructionObjectV3,
    StrategicReviewDirective,
)

logger = logging.getLogger(__name__)


async def handle_overview() -> List[mcp_types.PromptMessage]:
    """Provides a static description of the research loop."""
    operation = "handle_overview"
    logger.debug(f"Prompt '{operation}' accessed.")

    # Static description of the research loop
    research_loop_description = """
The Aya Tabular Research loop follows these general steps:
1.  **Define Task:** You provide the research goal, target data columns, and optional seeds using the `research_define_task` tool.
2.  **Initial Planning:** The server plans the first step, usually a DISCOVERY directive to find initial entities.
3.  **Execute Directive:** You receive a directive (InstructionObjectV3 or StrategicReviewDirective) and use your tools/knowledge to fulfill its requirements (e.g., find data, make a strategic decision).
4.  **Submit Report:** You submit your findings or decision using the `research_submit_inquiry_report` tool.
5.  **Process Report & Plan Next:** The server processes your report, updates its knowledge base, and plans the next directive (e.g., ENRICHMENT for specific entities, another DISCOVERY, or STRATEGIC_REVIEW).
6.  **Strategic Review:** Periodically, or when obstacles arise, the server issues a STRATEGIC_REVIEW directive, asking you to assess progress and decide the next phase (e.g., continue enrichment, discover more, finalize). User clarification might be requested via `research_submit_user_clarification`.
7.  **Repeat/Complete:** Steps 3-6 repeat until the research goal is met, criteria are satisfied, or you decide to finalize during a strategic review.
8.  **Results:** Once complete, you can preview results with `research_preview_results` and export the final dataset using `research_export_results`.
"""
    try:
        return [
            mcp_types.PromptMessage(
                role="system", content=research_loop_description.strip()
            )
        ]
    except Exception as e:
        logger.error(f"Error generating overview prompt: {e}", exc_info=True)
        # Return a simple error message within the prompt structure
        return [
            mcp_types.PromptMessage(
                role="system", content=f"Error generating overview: {e}"
            )
        ]
