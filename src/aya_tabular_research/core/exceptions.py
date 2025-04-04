"""
Custom exceptions for the AYA Guided Inquiry Framework server.

These exceptions carry structured data payloads (defined in models.error_models)
to provide detailed context for MCP error responses.
"""

from typing import Optional

from mcp.types import INTERNAL_ERROR, INVALID_REQUEST  # Standard MCP error codes
from pydantic import BaseModel

from .models.error_models import (
    OperationalErrorData,
    ValidationErrorData,
)


class AYAServerError(Exception):
    """Base class for custom server exceptions."""

    def __init__(
        self,
        message: str,
        code: int = INTERNAL_ERROR,
        data: Optional[BaseModel] = None,
    ):
        """
        Initializes the base server error.

        Args:
            message: A human-readable error message.
            code: The standard MCP error code (e.g., INTERNAL_ERROR, INVALID_REQUEST).
            data: An optional Pydantic model containing structured error details.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data


class TaskValidationError(AYAServerError):
    """Exception for errors during TaskDefinition validation."""

    def __init__(self, message: str, validation_data: ValidationErrorData):
        super().__init__(message, code=INVALID_REQUEST, data=validation_data)


class PlanningError(AYAServerError):
    """Exception for errors occurring during the planning phase."""

    def __init__(self, message: str, operation_data: OperationalErrorData):
        # Planning errors are typically internal server issues
        super().__init__(message, code=INTERNAL_ERROR, data=operation_data)


class KBInteractionError(AYAServerError):
    """Exception for errors during interaction with the Knowledge Base."""

    def __init__(self, message: str, operation_data: OperationalErrorData):
        # KB errors are typically internal server issues
        super().__init__(message, code=INTERNAL_ERROR, data=operation_data)
        # If KBInteractionErrorData is defined, use it instead:
        # super().__init__(message, code=INTERNAL_ERROR, data=kb_interaction_data)


class ReportProcessingError(AYAServerError):
    """Exception for errors during InquiryReport processing or validation."""

    def __init__(
        self,
        message: str,
        report_data: OperationalErrorData,
        code: int = INVALID_REQUEST,
    ):
        # Report processing errors can be due to invalid client input (INVALID_REQUEST)
        # or internal issues (INTERNAL_ERROR). Defaulting to INVALID_REQUEST.
        super().__init__(message, code=code, data=report_data)
        # If ReportProcessingErrorData is defined, use it instead:
        # super().__init__(message, code=code, data=report_processing_data)


# Add more specific exceptions as needed, e.g.:
# class StateTransitionError(AYAServerError): ...
