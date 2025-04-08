from enum import Enum


class OverallStatus(str, Enum):
    """Represents the overall status of the research task (Phase 2+)."""

    # --- Phase 1 States ---
    AWAITING_TASK_DEFINITION = "AWAITING_TASK_DEFINITION"
    AWAITING_DIRECTIVE = "AWAITING_DIRECTIVE"  # Ready to provide next directive

    # --- Phase 2+ States ---
    CONDUCTING_INQUIRY = "CONDUCTING_INQUIRY"  # Directive issued, awaiting report
    AWAITING_USER_CLARIFICATION = (
        "AWAITING_USER_CLARIFICATION"  # Blocked, needs user input (Phase 3)
    )

    # --- Terminal States ---
    RESEARCH_COMPLETE = "RESEARCH_COMPLETE"  # Task considered finished
    FATAL_ERROR = "FATAL_ERROR"  # Unrecoverable server error


class InquiryStatus(str, Enum):
    """Status reported by the client for a specific inquiry cycle."""

    COMPLETED = "COMPLETED"  # Inquiry finished successfully
    BLOCKED = "BLOCKED"  # Inquiry could not be completed due to an obstacle
    PARTIAL = "PARTIAL"  # Inquiry partially completed, more work possible/needed
    FAILED = "FAILED"  # Inquiry execution failed due to client-side error


# Add other enums as needed, e.g., ResearchPhase, CycleType


class DirectiveType(str, Enum):
    """Defines the different types of directives the server can issue."""

    DISCOVERY = "DISCOVERY"
    ENRICHMENT = "ENRICHMENT"
    CLARIFICATION = "CLARIFICATION"
    STRATEGIC_REVIEW = "STRATEGIC_REVIEW"
    COMPLETION = "COMPLETION"


class StrategicDecisionOption(str, Enum):
    """Valid strategic decisions the client can make during a review."""

    FINALIZE = "FINALIZE"
    DISCOVER = "DISCOVER"
    ENRICH = "ENRICH"
    ENRICH_SPECIFIC = "ENRICH_SPECIFIC"
    CLARIFY_USER = "CLARIFY_USER"
