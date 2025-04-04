from enum import Enum

class OverallStatus(str, Enum):
    """Represents the overall status of the research task (Phase 2+)."""
    # --- Phase 1 States ---
    AWAITING_TASK_DEFINITION = "AWAITING_TASK_DEFINITION" # Initial state
    # DEFINING_TASK = "DEFINING_TASK"                 # Internal processing, not a stable state
    # PLANNING_INITIAL = "PLANNING_INITIAL"             # Internal processing, not a stable state
    AWAITING_DIRECTIVE = "AWAITING_DIRECTIVE"         # Ready to provide next directive (or first)

    # --- Phase 2+ States ---
    CONDUCTING_INQUIRY = "CONDUCTING_INQUIRY"         # Directive issued, awaiting report
    AWAITING_USER_CLARIFICATION = "AWAITING_USER_CLARIFICATION" # Blocked, needs user input (Phase 3)
    # PLANNING_NEXT = "PLANNING_NEXT"                 # Internal processing after report, before AWAITING_DIRECTIVE

    # --- Terminal States ---
    RESEARCH_COMPLETE = "RESEARCH_COMPLETE"           # Task considered finished
    FATAL_ERROR = "FATAL_ERROR"                     # Unrecoverable server error


class InquiryStatus(str, Enum):
    """Status reported by the client for a specific inquiry cycle."""
    COMPLETED = "COMPLETED" # Inquiry finished successfully
    BLOCKED = "BLOCKED"     # Inquiry could not be completed due to an obstacle
    PARTIAL = "PARTIAL"     # Inquiry partially completed, more work possible/needed
    FAILED = "FAILED"       # Inquiry execution failed due to client-side error

# Add other enums as needed, e.g., ResearchPhase, CycleType