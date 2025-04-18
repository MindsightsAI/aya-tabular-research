import logging

# Consolidate typing imports and add Union
from typing import Any, Dict, Union

# Import custom exceptions and error models
from ..core.exceptions import KBInteractionError, PlanningError

# Import necessary models and types
from ..core.models.enums import DirectiveType, StrategicDecisionOption  # Import Enums
from ..core.models.error_models import OperationalErrorData
from ..core.models.research_models import StrategicReviewContext  # Added
from ..core.models.research_models import StrategicReviewDirective  # Added
from ..core.models.research_models import (
    InstructionObjectV3,
    ReportingGuideline,
    TaskDefinition,
)

# Import the refined PlannerSignal type and content types
from ..planning.planner import (  # Updated imports
    DiscoveryDirectiveContent,
    EnrichmentDirectiveContent,
    PlannerSignal,
)
from ..storage.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

MAX_PROFILE_ROWS = 100  # Configurable limit for full profile context
# Define type alias for the builder result (Now returns only the directive)
BuilderResult = Union[InstructionObjectV3, StrategicReviewDirective]


# Renaming class for clarity
class DirectiveBuilder:
    """
    Constructs the appropriate Directive (InstructionObjectV3 or StrategicReviewDirective)
    based on the Planner's signal and retrieves necessary context from the Knowledge Base.
    Raises PlanningError if context retrieval fails.
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        if not isinstance(knowledge_base, KnowledgeBase):
            raise TypeError("knowledge_base must be an instance of KnowledgeBase")
        self._kb = knowledge_base
        logger.info("DirectiveBuilder initialized.")

    def build_directive(
        self,
        planner_signal: PlannerSignal,
        task_definition: TaskDefinition,
    ) -> BuilderResult:  # Updated return type hint
        """
        Creates the appropriate Directive object, embedding context data directly.

        Args:
            planner_signal: The output signal from the Planner.
            task_definition: The overall task definition for context.

        Returns:
            The constructed Directive instance (InstructionObjectV3 or StrategicReviewDirective)
            with context data embedded within it.

        Raises:
            PlanningError: If required context cannot be retrieved from the KnowledgeBase.
            ValueError: If an unexpected planner signal is received.
        """
        directive: Union[InstructionObjectV3, StrategicReviewDirective]

        signal_type = planner_signal[0]
        signal_payload = planner_signal[1]
        logger.debug(
            f"build_directive: Received planner signal type: {signal_type}, payload: {signal_payload}"
        )

        try:  # Wrap the entire building process to catch KB errors
            if signal_type == DirectiveType.ENRICHMENT:  # Use Enum
                # --- Build Enrichment Directive ---
                enrichment_content: EnrichmentDirectiveContent = signal_payload
                inquiry_focus, focus_areas, target_entity_id = enrichment_content
                logger.debug(
                    f"Building ENRICH directive for target: {target_entity_id}"
                )

                # Fetch entity profile for context
                logger.debug(
                    f"Fetching profile for entity '{target_entity_id}' for enrichment context."
                )
                # Fetch the full entity profile as per architectural plan
                full_entity_profile = self._kb.get_entity_profile(
                    target_entity_id
                )  # Might raise KBInteractionError
                if not full_entity_profile:
                    logger.warning(
                        f"Could not retrieve profile for entity '{target_entity_id}'. directive_context will be None."
                    )
                    context_to_send = None  # Initialize context to send
                elif (
                    isinstance(full_entity_profile, list)
                    and len(full_entity_profile) > MAX_PROFILE_ROWS
                ):
                    logger.warning(
                        f"Entity profile for '{target_entity_id}' exceeds {MAX_PROFILE_ROWS} rows. Sending summary context."
                    )
                    # Create a summary (example: first row + count)
                    # TODO: Implement more sophisticated summarization
                    summary_context = {
                        "summary_type": "truncated_profile",
                        "total_rows": len(full_entity_profile),
                        "first_row_preview": (
                            full_entity_profile[0] if full_entity_profile else None
                        ),
                        "message": f"Full profile truncated. Only showing first row out of {len(full_entity_profile)}.",
                    }
                    context_to_send = summary_context
                else:
                    # Profile is within limits or not a list (should be dict or None)
                    context_to_send = full_entity_profile

                required_schema_points = focus_areas
                reporting_guidelines = ReportingGuideline(
                    required_schema_points=required_schema_points,
                    narrative_summary_required=True,
                )
                logger.debug(
                    f"Set enrichment reporting guidelines, required points: {required_schema_points}"
                )

                directive = InstructionObjectV3(
                    research_goal_context=task_definition.task_description,
                    inquiry_focus=inquiry_focus,
                    focus_areas=focus_areas,
                    reporting_guidelines=reporting_guidelines,
                    allowed_tools=["research_submit_inquiry_report"],
                    target_entity_id=target_entity_id,
                    directive_type=DirectiveType.ENRICHMENT,  # Use Enum
                    directive_context=context_to_send,  # Embed full profile or summary context
                )
                logger.info(
                    f"Built ENRICH instruction {directive.instruction_id} for focus: '{inquiry_focus}'"
                )

            elif signal_type == DirectiveType.DISCOVERY:  # Use Enum
                # --- Build Discovery Directive ---
                discovery_content: DiscoveryDirectiveContent = signal_payload
                inquiry_focus, focus_areas, _ = discovery_content  # target_id is None
                logger.debug("Building DISCOVERY directive.")

                required_schema_points = [task_definition.identifier_column]
                reporting_guidelines = ReportingGuideline(
                    required_schema_points=required_schema_points,
                    narrative_summary_required=False,
                )
                logger.debug(
                    f"Set discovery reporting guidelines, required points: {required_schema_points}"
                )

                directive = InstructionObjectV3(
                    research_goal_context=task_definition.task_description,
                    inquiry_focus=inquiry_focus,
                    focus_areas=focus_areas,
                    reporting_guidelines=reporting_guidelines,
                    allowed_tools=["research_submit_inquiry_report"],
                    target_entity_id=None,
                    directive_type=DirectiveType.DISCOVERY,  # Use Enum
                    # No directive_context needed for discovery
                )
                logger.info(
                    f"Built DISCOVERY instruction {directive.instruction_id} for focus: '{inquiry_focus}'"
                )

            elif signal_type == DirectiveType.STRATEGIC_REVIEW:  # Use Enum
                # --- Build Strategic Review Directive ---
                # Expect payload to be a dictionary now
                review_payload: Dict[str, Any] = signal_payload
                review_reason = review_payload.get("reason", "Unknown reason")
                enrichment_cycle_count = review_payload.get("enrichment_cycle_count")
                completeness_ratio = review_payload.get("completeness_ratio")
                planner_suggestion = review_payload.get("planner_suggestion")

                logger.debug(
                    f"Building STRATEGIC_REVIEW directive. Reason: {review_reason}, Cycle: {enrichment_cycle_count}, Completeness: {completeness_ratio}, Suggestion: {planner_suggestion}"
                )

                # Gather context - these might raise KBInteractionError
                kb_summary = self._kb.get_knowledge_summary()
                # Always fetch obstacle summary for strategic review context
                obstacle_summary = self._kb.get_obstacle_summary()
                logger.debug(
                    f"Obstacle summary for strategic review context: {obstacle_summary}"
                )
                # Fetch incomplete entities list as per architectural plan
                incomplete_entities = self._kb.find_incomplete_entities(
                    task_definition.target_columns
                )
                logger.debug(
                    f"Incomplete entities for strategic review context: {incomplete_entities}"
                )
                strategic_context_data = StrategicReviewContext(
                    review_reason=review_reason,
                    research_goal=task_definition.task_description,
                    kb_summary=kb_summary,
                    obstacle_summary=obstacle_summary,
                    incomplete_entities=incomplete_entities,
                    # Add new context fields
                    enrichment_cycle_count=enrichment_cycle_count,
                    completeness_ratio=completeness_ratio,
                    planner_suggestion=planner_suggestion,
                )

                # Dynamically generate the options string from the Enum
                options_str = ", ".join(
                    [option.value for option in StrategicDecisionOption]
                )
                focus_areas_review = [
                    f"Strategic Review Triggered: {review_reason}",
                    "Review the embedded 'strategic_context' (KB Summary, Obstacles, Incomplete Entities).",
                    "Assess the overall research status against the 'research_goal'.",
                    f"Decide the next strategic phase by setting the 'strategic_decision' field in your report to one of: {options_str}.",
                    f"If choosing '{StrategicDecisionOption.ENRICH_SPECIFIC.value}', also provide a list of entity IDs in 'strategic_targets'.",
                ]

                directive = StrategicReviewDirective(
                    research_goal_context=task_definition.task_description,
                    review_reason=review_reason,
                    focus_areas=focus_areas_review,
                    strategic_context=strategic_context_data,  # Embed context object here
                )
                logger.info(
                    f"Built STRATEGIC_REVIEW instruction {directive.directive_id}. Reason: {review_reason}"
                )

            else:
                # Should not happen if PlannerSignal is handled correctly
                msg = f"Received unexpected planner signal type: {signal_type}. Cannot build directive."
                # Log the unexpected signal details clearly before raising
                logger.error(
                    f"build_directive: {msg} Signal Type: {signal_type}, Payload: {signal_payload}"
                )
                # Raise ValueError for unexpected input type
                raise ValueError(msg)

            # Return only the built directive (context is embedded)
            return directive

        except KBInteractionError as e:
            # Catch specific KB errors during context retrieval
            logger.error(
                f"Failed to retrieve context from KnowledgeBase during directive building: {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="DirectiveBuilder",
                operation=f"build_{signal_type.lower()}_directive",  # Be more specific
                details=f"Failed KB interaction: {e.message}",
                internal_code=getattr(
                    e.data, "internal_code", None
                ),  # Propagate internal code if present
            )
            # Raise PlanningError as the builder failed due to KB issues
            raise PlanningError(
                f"Failed to build directive due to KB error: {e}",
                operation_data=op_data,
            ) from e
        except Exception as e:
            # Catch any other unexpected errors during building
            logger.error(
                f"Unexpected error building directive for signal {signal_type}: {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="DirectiveBuilder",
                operation=f"build_{signal_type.lower()}_directive",
                details=f"Unexpected error: {e}",
            )
            # Raise PlanningError for unexpected builder failures
            raise PlanningError(
                f"Unexpected error building directive: {e}", operation_data=op_data
            ) from e
