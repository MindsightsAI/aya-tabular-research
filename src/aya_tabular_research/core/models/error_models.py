"""
Pydantic models for structured error data payloads used in MCP ErrorData.
"""
from typing import Any, List, Optional

from pydantic import BaseModel


class FieldErrorDetail(BaseModel):
    """Details about an error related to a specific input field."""

    field: str = "unknown"  # Field name or path (e.g., "task_definition.columns")
    error: str  # Description of the error
    value: Optional[Any] = None  # The problematic value, if available


class ValidationErrorData(BaseModel):
    """Structured data for validation errors (MCP code: INVALID_REQUEST)."""

    detail: List[FieldErrorDetail]
    suggestion: Optional[str] = None  # Optional hint for the client/user


class OperationalErrorData(BaseModel):
    """Structured data for internal server operational errors (MCP code: INTERNAL_ERROR)."""

    component: str  # e.g., "Planner", "KnowledgeBase", "StateManager"
    operation: str  # e.g., "determine_next_directive", "get_entity_profile"
    is_retryable: bool = False
    internal_code: Optional[str] = None  # Optional internal code for detailed logging/lookup
    details: Optional[str] = None # Optional extra context about the error


# Add more specific error data models here as needed, e.g.:
# class KBInteractionErrorData(OperationalErrorData): ...
# class ReportProcessingErrorData(OperationalErrorData): ...