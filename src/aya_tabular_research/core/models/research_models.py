import uuid
from enum import Enum  # Added import
from typing import Any, Dict, List, Literal, Optional, Union  # Added Union

from pydantic import BaseModel, Field, field_validator

# Import necessary enums
from .enums import InquiryStatus

# --- Task Definition ---


class TaskDefinition(BaseModel):  # Keep original TaskDefinition
    """Defines the overall research task."""

    task_description: str = Field(
        ..., description="Natural language goal of the research."
    )
    columns: List[str] = Field(
        ..., description="List of all desired column names for the final dataset."
    )
    identifier_column: str = Field(
        ..., description="The column name that uniquely identifies each entity/row."
    )
    target_columns: List[str] = Field(
        default_factory=list,
        description="Columns that require active enrichment after initial discovery.",
    )
    seed_entities: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Optional list of initial entities (as dicts) to seed the knowledge base.",
    )
    hints: Optional[str] = Field(
        None, description="Optional hints or context for the discovery process."
    )
    # Add other fields like query_guidance, context_columns later if needed

    @field_validator("columns")
    @classmethod
    def validate_columns_unique_and_not_empty(cls, v):
        if not v:
            raise ValueError("Columns list cannot be empty.")
        if len(v) != len(set(v)):
            raise ValueError("Column names must be unique.")
        reserved = {
            c for c in v if c.startswith("_")
        }  # Basic check for internal prefixes
        if reserved:
            raise ValueError(f"Columns cannot start with underscore: {reserved}")
        return v

    @field_validator("identifier_column")
    @classmethod
    def validate_identifier_in_columns(cls, v, info: Any):
        cols = info.data.get("columns", [])
        if v not in cols:
            raise ValueError(
                f"Identifier column '{v}' must be present in the columns list."
            )
        return v

    @field_validator("target_columns")
    @classmethod
    def validate_targets_in_columns_and_not_identifier(cls, v, info: Any):
        cols = set(info.data.get("columns", []))
        id_col = info.data.get("identifier_column")
        invalid_targets = [t for t in v if t not in cols]
        if invalid_targets:
            raise ValueError(
                f"Target columns must be present in the columns list: {invalid_targets}"
            )
        if id_col and id_col in v:
            raise ValueError(f"Identifier column '{id_col}' cannot be a target column.")
        return v


# --- MCP Tool Arguments ---


class DefineTaskArgs(BaseModel):
    """Arguments for the research/define_task MCP tool."""

    task_definition: TaskDefinition


# --- Instruction Object ---


class ReportingGuideline(BaseModel):
    """Specifies how the client should report findings."""

    required_schema_points: List[str] = Field(
        default_factory=list,
        description="Specific keys required within structured_data_points.",
    )
    narrative_summary_required: bool = Field(
        default=True, description="Whether a narrative summary is expected."
    )
    # Add more guidelines later (e.g., confidence score requirements)


class InstructionObjectV3(BaseModel):
    """
    Represents the action specification part of a directive sent to the client.
    Contextual data is passed separately via MCP resources.
    """

    instruction_id: str = Field(default_factory=lambda: f"instr_{uuid.uuid4().hex[:8]}")
    research_goal_context: str = Field(
        ..., description="Brief reminder of the overall research goal."
    )
    inquiry_focus: str = Field(
        ...,
        description="Specifies the target/action for this cycle (e.g., 'Action: Discover initial entities...', 'Action: Enrich entity \\'XYZ\\'').",
    )
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Specific attributes, questions, or context (like task description for discovery) to prioritize.",
    )
    reporting_guidelines: ReportingGuideline = Field(
        default_factory=ReportingGuideline,
        description="Instructions on how the client should structure the InquiryReport.",
    )
    allowed_tools: List[str] = Field(
        default=["research/submit_inquiry_report"],
        description="MCP tools the client is allowed to use while executing this instruction.",
    )
    directive_type: Literal[
        "DISCOVERY", "ENRICHMENT", "STRATEGIC_REVIEW", "COMPLETION"
    ] = Field(
        ...,
        description="The type of directive (Discovery, Enrichment, Strategic Review, or Completion).",
    )
    target_entity_id: Optional[str] = Field(
        None,
        description="The specific entity ID this instruction targets (for enrichment directives).",
    )
    # --- NEW Field for Embedded Context ---
    directive_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Context data relevant to the directive, typically the entity profile for ENRICHMENT directives.",
    )

    # --- Placeholder/Helper Methods ---

    @classmethod
    def placeholder_planning_complete(cls) -> "InstructionObjectV3":
        # This placeholder might need adjustment or removal depending on final completion flow
        return cls(
            research_goal_context="N/A",
            inquiry_focus="Research Complete",
            allowed_tools=["research/preview_results", "research/export_results"],
        )

    @classmethod
    def placeholder_fatal_error(cls, message: str) -> "InstructionObjectV3":
        return cls(
            research_goal_context="N/A - Fatal Error",
            inquiry_focus=f"Fatal Error: {message}",
            allowed_tools=[],  # No tools allowed in fatal error state
        )


# --- Strategic Review Models (Proposal 6.3) ---


class StrategicReviewContext(BaseModel):
    """Context provided to the client for a strategic review."""

    review_reason: str = Field(
        ...,
        description="Reason why the strategic review was triggered (e.g., 'enrichment_complete', 'critical_obstacles').",
    )
    research_goal: str = Field(
        ..., description="The overall research goal description."
    )
    kb_summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics from KnowledgeBase.get_knowledge_summary().",
    )
    obstacle_summary: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Summary of entities with active obstacles (e.g., [{'entity_id': 'X', 'obstacle': '...'}, ...]). Provided if reason is 'critical_obstacles'.",
    )
    # Add other context fields as needed


class StrategicReviewDirective(BaseModel):
    """Directive specifically requesting a strategic review from the client."""

    directive_id: str = Field(default_factory=lambda: f"srev_{uuid.uuid4().hex[:8]}")
    directive_type: Literal["STRATEGIC_REVIEW"] = Field("STRATEGIC_REVIEW", frozen=True)
    research_goal_context: str  # Copied from TaskDefinition
    review_reason: str
    focus_areas: List[str] = Field(
        ...,
        description="Instructions for the client, e.g., 'Review context in resource', 'Assess goal completion', 'Decide next phase: FINALIZE, DISCOVER, ENRICH, ENRICH_SPECIFIC, CLARIFY_USER'.",
    )
    reporting_guidelines: Dict[str, Any] = Field(
        default={"required_report_field": "strategic_decision"},
        description="Specifies that the client MUST provide the 'strategic_decision' field in the InquiryReport.",
    )
    allowed_tools: List[str] = Field(default=["research/submit_inquiry_report"])
    # --- NEW Field for Embedded Context ---
    strategic_context: Optional["StrategicReviewContext"] = (
        Field(  # Use forward reference string
            default=None,
            description="Detailed context provided for the strategic review, based on the StrategicReviewContext model.",
        )
    )


# --- End Strategic Review Models ---


# --- Inquiry Report ---


class SynthesizedFinding(BaseModel):
    """Represents a synthesized finding or insight from the client."""

    finding: str = Field(..., description="The textual description of the finding.")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Client's confidence in the finding (0.0-1.0).",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence (e.g., source URIs, text snippets).",
    )


class IdentifiedObstacle(BaseModel):
    """Represents an obstacle encountered by the client."""

    obstacle: str = Field(..., description="Description of the obstacle.")
    details: Optional[str] = Field(
        None, description="Additional details about the obstacle."
    )


class ProposedNextStep(BaseModel):
    """Represents a next step proposed by the client."""

    proposal: str = Field(..., description="Description of the proposed next step.")
    rationale: Optional[str] = Field(None, description="Reasoning behind the proposal.")


class InquiryReport(BaseModel):
    """
    Represents the report submitted by the client after completing an inquiry cycle.
    Includes fields for richer reporting (Phase 3+).
    """

    instruction_id: str = Field(
        ..., description="The ID of the instruction this report corresponds to."
    )
    status: InquiryStatus = Field(
        ..., description="Status of the inquiry execution (e.g., COMPLETED, BLOCKED)."
    )
    structured_data_points: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of dictionaries containing structured findings (entities/attributes). Format depends on reporting_guidelines.",
    )
    summary_narrative: Optional[str] = Field(
        None, description="Optional narrative summary of the findings for this cycle."
    )
    # --- Phase 3+ Fields ---
    synthesized_findings: Optional[List[SynthesizedFinding]] = Field(
        default_factory=list, description="List of synthesized insights or findings."
    )
    identified_obstacles: Optional[List[IdentifiedObstacle]] = Field(
        default_factory=list,
        description="List of obstacles encountered during execution.",
    )
    proposed_next_steps: Optional[List[ProposedNextStep]] = Field(
        default_factory=list, description="List of next steps proposed by the client."
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the reported data/findings in this cycle (0.0-1.0).",
    )
    # --- End Phase 3+ Fields ---

    # --- Fields for Proposal 6.3 ---
    strategic_decision: Optional[
        Literal["FINALIZE", "DISCOVER", "ENRICH", "ENRICH_SPECIFIC", "CLARIFY_USER"]
    ] = Field(
        None,
        description="Client's strategic decision in response to a STRATEGIC_REVIEW directive.",
    )
    strategic_targets: Optional[List[str]] = Field(
        None,
        description="Optional list of entity IDs to prioritize if strategic_decision is ENRICH_SPECIFIC.",
    )
    # --- End Fields for Proposal 6.3 ---

    # Add validator to ensure structured_data_points contain identifier? Maybe later.


# --- MCP Tool Arguments ---


class SubmitInquiryReportArgs(BaseModel):
    """Arguments for the research/submit_inquiry_report MCP tool."""

    inquiry_report: InquiryReport


# --- NEW MODEL for Phase 3 ---
class SubmitUserClarificationArgs(BaseModel):
    """Arguments for the research/submit_user_clarification MCP tool."""

    # Simple text clarification for now, could be more structured later
    clarification: Optional[str] = Field(
        None, description="The clarification or decision provided by the user."
    )
    # Potentially add fields like 'decision_type' or structured choices later


# --- MCP Tool Results (Specific to this server) ---


class RequestDirectiveResultStatus(str, Enum):
    """Possible statuses for the result of requesting a directive."""

    DIRECTIVE_ISSUED = "directive_issued"
    CLARIFICATION_NEEDED = "clarification_needed"
    RESEARCH_COMPLETE = "research_complete"
    ERROR = "error"


class MCPResourceRepresentation(BaseModel):
    """Mimics the structure of an MCP resource block for embedding."""

    uri: str
    text: str
    mimeType: str = "application/json"


class MCPEmbeddedResourceRepresentation(BaseModel):
    """Mimics the structure of an MCP EmbeddedResource for tool results."""

    type: str = "resource"
    resource: MCPResourceRepresentation


# RequestDirectiveResult is removed as its fields are merged into other results

# --- MCP Resource Payloads ---


class ServerStatusPayload(BaseModel):
    """Payload for the server status resource."""

    status: str = Field(..., description="The current OverallStatus value.")
    available_tools: List[str] = Field(
        ..., description="List of tool names available in the current state."
    )
    task_defined: bool = Field(
        ..., description="Whether a task definition has been loaded."
    )
    active_instruction_id: Optional[str] = Field(
        None, description="ID of the currently active instruction, if any."
    )


# --- Enhanced Tool Results (Including Status) ---


class DefineTaskResult(BaseModel):
    """Result for the research/define_task tool, including the first directive."""

    message: str = Field(..., description="Confirmation message for task definition.")
    # Fields from RequestDirectiveResult representing the *first* directive/status
    status: RequestDirectiveResultStatus = Field(
        ...,
        description="Indicates the outcome of the initial planning (directive issued, clarification needed, etc.).",
    )
    summary: str = Field(
        ..., description="A brief natural language summary of the first step or status."
    )
    # Use Union for the actual payload object, exclude from direct serialization
    instruction_payload: Optional[
        Union[InstructionObjectV3, StrategicReviewDirective]
    ] = Field(
        None,
        description="The first directive object (InstructionObjectV3 or StrategicReviewDirective) including embedded context.",
    )
    error_message: Optional[str] = Field(
        None, description="Present only if initial planning resulted in an error."
    )


class SubmitReportAndGetDirectiveResult(BaseModel):
    """Combined result for research/submit_inquiry_report, including the next directive."""

    message: str = Field(..., description="Confirmation message for report submission.")
    # Fields from RequestDirectiveResult representing the *next* directive/status
    status: RequestDirectiveResultStatus = Field(
        ...,
        description="Indicates the outcome of the planning after the report (directive issued, clarification needed, etc.).",
    )
    summary: str = Field(
        ..., description="A brief natural language summary of the next step or status."
    )
    # Use Union for the actual payload object, exclude from direct serialization
    instruction_payload: Optional[
        Union[InstructionObjectV3, StrategicReviewDirective]
    ] = Field(
        None,
        description="The next directive object (InstructionObjectV3 or StrategicReviewDirective) including embedded context.",
    )
    error_message: Optional[str] = Field(
        None,
        description="Present only if planning after the report resulted in an error.",
    )


class SubmitClarificationResult(BaseModel):
    """Result for research/submit_user_clarification, including the next directive."""

    message: str = Field(
        ..., description="Confirmation message for clarification submission."
    )
    # Fields representing the *next* directive/status after clarification
    status: RequestDirectiveResultStatus = Field(
        ...,
        description="Indicates the outcome of the planning after clarification (directive issued, etc.).",
    )
    summary: str = Field(
        ..., description="A brief natural language summary of the next step or status."
    )
    # Use Union for the actual payload object, exclude from direct serialization
    instruction_payload: Optional[
        Union[InstructionObjectV3, StrategicReviewDirective]
    ] = Field(
        None,
        description="The next directive object (InstructionObjectV3 or StrategicReviewDirective) including embedded context.",
    )
    error_message: Optional[str] = Field(
        None,
        description="Present only if planning after clarification resulted in an error.",
    )
    new_status: str = Field(
        ..., description="The server status after processing the clarification."
    )
    available_tools: List[str] = Field(
        ..., description="Tools available in the new state."
    )


class ExportFormat(str, Enum):
    """Supported formats for data export."""

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class ExportResultsArgs(BaseModel):
    """Arguments for the research/export_results MCP tool."""

    format: ExportFormat = Field(
        default=ExportFormat.CSV,
        description="The desired file format for the export (csv, json, or parquet).",
    )


class ExportResult(BaseModel):
    """Result payload for the research/export_results MCP tool."""

    file_path: str = Field(
        ..., description="The server-side path to the exported file."
    )
