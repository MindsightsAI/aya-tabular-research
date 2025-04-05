# Central registry for shared component instances
# Populated during application startup (e.g., in mcp_interface.py)
# Imported by handlers to access shared state/functionality.

from typing import TYPE_CHECKING, Optional

# Import types for hinting only to avoid new circular dependencies
if TYPE_CHECKING:
    from ..execution.instruction_builder import DirectiveBuilder
    from ..execution.report_handler import ReportHandler
    from ..planning.planner import Planner
    from ..storage.knowledge_base import KnowledgeBase
    from .state_manager import StateManager

# Define placeholders with type hints
state_manager: Optional["StateManager"] = None
planner: Optional["Planner"] = None
directive_builder: Optional["DirectiveBuilder"] = None
report_handler: Optional["ReportHandler"] = None
knowledge_base: Optional["KnowledgeBase"] = None