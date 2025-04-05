import logging

# Consolidate typing imports and add Union
from typing import Union

# Import custom exceptions and error models
from ..core.exceptions import KBInteractionError, PlanningError

# Import necessary models and types
from ..core.models.enums import DirectiveType  # Import the new Enum
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

        try:  # Wrap the entire building process to catch KB errors
            if signal_type == "ENRICH":
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
                directive_context_data = self._kb.get_entity_profile(
                    target_entity_id
                )  # Might raise KBInteractionError
                if not directive_context_data:
                    logger.warning(
                        f"Could not retrieve profile for entity '{target_entity_id}'. directive_context will be None."
                    )

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
                    allowed_tools=["research/submit_inquiry_report"],
                    target_entity_id=target_entity_id,
                    directive_type=DirectiveType.ENRICHMENT,  # Use Enum
                    directive_context=directive_context_data,  # Embed context here
                )
                logger.info(
                    f"Built ENRICH instruction {directive.instruction_id} for focus: '{inquiry_focus}'"
                )

            elif signal_type == "DISCOVERY":
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
                    allowed_tools=["research/submit_inquiry_report"],
                    target_entity_id=None,
                    directive_type=DirectiveType.DISCOVERY,  # Use Enum
                    # No directive_context needed for discovery
                )
                logger.info(
                    f"Built DISCOVERY instruction {directive.instruction_id} for focus: '{inquiry_focus}'"
                )

            elif signal_type == "NEEDS_STRATEGIC_REVIEW":
                # --- Build Strategic Review Directive ---
                review_reason: str = signal_payload
                logger.debug(
                    f"Building STRATEGIC_REVIEW directive. Reason: {review_reason}"
                )

                # Gather context - these might raise KBInteractionError
                kb_summary = self._kb.get_knowledge_summary()
                # Always fetch obstacle summary for strategic review context
                obstacle_summary = self._kb.get_obstacle_summary()
                logger.debug(
                    f"Obstacle summary for strategic review context: {obstacle_summary}"
                )
                strategic_context_data = StrategicReviewContext(
                    review_reason=review_reason,
                    research_goal=task_definition.task_description,
                    kb_summary=kb_summary,
                    obstacle_summary=obstacle_summary,
                )

                focus_areas_review = [
                    f"Strategic Review Triggered: {review_reason}",
                    "Review the context provided in the associated resource (KB Summary, Obstacles).",
                    "Assess the overall research status against the goal.",
                    "Decide the next strategic phase and report it using the 'strategic_decision' field.",
                    "Options: FINALIZE, DISCOVER, ENRICH, ENRICH_SPECIFIC, CLARIFY_USER.",
                    "If choosing ENRICH_SPECIFIC, also provide 'strategic_targets'.",
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
                logger.error(msg)
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
