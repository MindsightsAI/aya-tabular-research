import logging
from typing import Union

from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_REQUEST, ErrorData
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

# Import custom exceptions and error models from the correct relative path
from ..core.exceptions import AYAServerError
from ..core.models.error_models import (
    FieldErrorDetail,
    OperationalErrorData,
    ValidationErrorData,
)

logger = logging.getLogger(__name__)


def handle_aya_exception(e: Exception, operation: str) -> McpError:
    """Converts AYAServerError, PydanticValidationError, or generic Exception into McpError."""
    if isinstance(e, AYAServerError):
        logger.error(
            f"AYA Server Error during {operation}: {e.message}",
            exc_info=True,  # Log traceback for server errors too
        )
        # Ensure data is serializable (Pydantic models are handled, others passed as-is)
        data_payload: Union[dict, str, None] = None
        if isinstance(e.data, BaseModel):
            try:
                data_payload = e.data.model_dump(mode="json")
            except Exception as dump_err:
                logger.warning(f"Could not dump error data model: {dump_err}")
                data_payload = {"error": "Failed to serialize error data"}
        elif isinstance(e.data, (dict, str)):
            data_payload = e.data
        elif e.data is not None:
            logger.warning(f"Non-serializable error data type: {type(e.data)}")
            data_payload = {
                "error": f"Non-serializable error data type: {type(e.data).__name__}"
            }

        # NOTE: We return the McpError here, the caller should raise it with 'from e'
        return McpError(ErrorData(code=e.code, message=e.message, data=data_payload))
    elif isinstance(e, PydanticValidationError):
        logger.warning(
            f"Pydantic Validation Error during {operation}: {e}", exc_info=False
        )
        details = [
            FieldErrorDetail(
                field=".".join(map(str, err.get("loc", ["unknown"]))),
                error=err.get("msg", "Unknown validation error"),
            )
            for err in e.errors()
        ]
        validation_data = ValidationErrorData(
            detail=details,
            suggestion="Check input arguments against the tool/model schema.",
        )
        # NOTE: We return the McpError here, the caller should raise it with 'from e'
        return McpError(
            ErrorData(
                code=INVALID_REQUEST,
                message="Input validation failed.",
                data=validation_data.model_dump(mode="json"),
            )
        )
    else:
        logger.critical(f"Unhandled exception during {operation}: {e}", exc_info=True)
        op_data = OperationalErrorData(
            component="MCPInterface",  # Or determine dynamically if possible
            operation=operation,
            details=f"Unexpected error: {type(e).__name__}",
        )
        # NOTE: We return the McpError here, the caller should raise it with 'from e'
        return McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message="An unexpected internal server error occurred.",
                data=op_data.model_dump(mode="json"),
            )
        )
