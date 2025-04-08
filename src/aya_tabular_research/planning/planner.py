import logging
from collections import defaultdict  # Added for proposal boosts
from typing import Any, Dict, List, Optional, Tuple, Union  # Removed Literal

import pandas as pd

# Import custom exceptions and error models
from ..core.exceptions import KBInteractionError, PlanningError
from ..core.models.enums import DirectiveType  # Import Enum
from ..core.models.error_models import OperationalErrorData
from ..core.models.research_models import TaskDefinition
from ..storage.knowledge_base import KnowledgeBase

# Import internal column constants from tabular_store
from ..storage.tabular_store import (  # FINDINGS_COL, # Removed as findings analysis is not implemented yet; NARRATIVE_COL, # Removed as narrative is not used in planning logic yet
    CONFIDENCE_COL,
    OBSTACLES_COL,
    PROPOSALS_COL,
)

logger = logging.getLogger(__name__)

# Type for standard Enrichment directive content: (inquiry_focus, focus_areas, target_entity_id)
EnrichmentDirectiveContent = Tuple[
    str, List[str], str
]  # Target ID is required for enrichment
# Type for standard Discovery directive content: (inquiry_focus, focus_areas, None)
DiscoveryDirectiveContent = Tuple[str, List[str], None]

# Type for the planner's output signal to the MCP Interface
# It indicates the *type* of action needed and provides necessary parameters.
PlannerSignal = Union[
    Tuple[DirectiveType.ENRICHMENT, EnrichmentDirectiveContent],  # Use Enum
    Tuple[DirectiveType.DISCOVERY, DiscoveryDirectiveContent],  # Use Enum
    Tuple[DirectiveType.STRATEGIC_REVIEW, str],  # Use Enum, includes reason
    DirectiveType.CLARIFICATION,  # Use Enum, if obstacles block all paths
    None,  # Indicates research completion
]

# Constants for prioritization logic

# Strategic review thresholds are now instance attributes (_obstacle_threshold, _confidence_threshold)

CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_BOOST = 2.0  # Use float
PROPOSAL_BOOST = 1.0  # Use float
# RECENCY_PENALTY_FACTOR = 0.1 # Example: Penalize recently attempted entities slightly (Not implemented yet)
RETRY_PRIORITY_BOOST = (
    100.0  # Very high boost for entities explicitly requested for retry
)
MENTIONED_IN_PROPOSAL_BOOST = (
    50.0  # Significant boost for entities mentioned in proposals
)


class CandidateEvaluation(Dict):
    """Helper class for type hinting candidate evaluation dictionaries."""

    id: str
    profile: Optional[Dict[str, Any]]
    priority: float
    missing_attributes: List[str]


class Planner:
    """
    Strategically determines the next inquiry directive for the LLM client.

    Analyzes the research goal (TaskDefinition) and the current state of the
    Knowledge Base (KB) to decide between discovering new entities or enriching
    existing ones. Incorporates feedback from previous reports (obstacles,
    proposals, confidence) to prioritize targets and adapt focus.
    Handles completion checks and requests for user clarification when blocked.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        obstacle_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        stagnation_cycles: int = 3,  # Add stagnation threshold
    ):
        """
        Initializes the Planner.

        Args:
            knowledge_base: An instance conforming to the KnowledgeBase interface.
            obstacle_threshold: Ratio of entities with obstacles to trigger review.
            confidence_threshold: Average confidence level below which to trigger review.
            stagnation_cycles: Number of cycles without change to trigger stagnation review.
        """
        if not isinstance(knowledge_base, KnowledgeBase):
            raise TypeError("knowledge_base must be an instance of KnowledgeBase")
        self._kb = knowledge_base
        self._task_definition: Optional[TaskDefinition] = None
        self._obstacle_threshold = obstacle_threshold
        self._confidence_threshold = confidence_threshold
        self._stagnation_cycles_threshold = stagnation_cycles
        # Stagnation tracking state
        self._stagnation_counter: int = 0
        self._last_incomplete_set: Optional[set[str]] = None
        # State for prioritizing retries after clarification
        self._entities_to_prioritize_retry: List[str] = []
        # State for tracking processed base entities during enrichment cycle
        self._processed_base_entities_enrichment: set[str] = set()
        logger.info(
            f"Planner initialized with Obstacle Threshold: {self._obstacle_threshold}, "
            f"Confidence Threshold: {self._confidence_threshold}, "
            f"Stagnation Cycles: {self._stagnation_cycles_threshold}"
        )

    def set_task_definition(self, task_definition: TaskDefinition):
        """Stores the task definition, providing context for planning."""
        if not isinstance(task_definition, TaskDefinition):
            raise TypeError("task_definition must be an instance of TaskDefinition")
        self._task_definition = task_definition
        logger.info("Task definition set in Planner.")

    def incorporate_clarification(self, clarification: Optional[str]):
        """
        Processes user clarification to potentially influence future planning.
        Specifically looks for requests to retry blocked entities and clears their obstacles.
        Populates self._entities_to_prioritize_retry for the next planning cycle.
        Raises PlanningError if KB interaction fails during processing.
        """
        # Reset retry list at the start of processing each clarification
        self._entities_to_prioritize_retry = []

        if not clarification:
            logger.info("Planner received empty clarification.")
            return

        logger.info(
            f"Planner received user clarification: '{clarification}'"
        )  # Log full text

        # Basic parsing for retry requests
        lower_clarification = clarification.lower()
        if "retry" in lower_clarification or "try again" in lower_clarification:
            logger.debug(
                "Clarification contains retry keywords. Attempting to identify entities."
            )
            try:
                all_entities = self._kb.get_all_entity_ids()
                logger.debug(f"All known entity IDs from KB: {all_entities}")
            except KBInteractionError as kbe:
                logger.error(
                    f"Failed to get entity IDs from KB during clarification processing: {kbe}",
                    exc_info=True,
                )
                raise PlanningError(
                    f"Failed to get entity IDs for clarification: {kbe.message}",
                    operation_data=kbe.data,
                ) from kbe
            except Exception as e:
                logger.error(
                    f"Unexpected error getting entity IDs from KB during clarification processing: {e}",
                    exc_info=True,
                )
                raise PlanningError(
                    f"Unexpected error getting entity IDs for clarification: {e}"
                ) from e

            # Identify entities mentioned in the clarification (case-insensitive check)
            entities_mentioned = [
                entity_id
                for entity_id in all_entities
                if entity_id.lower() in lower_clarification
            ]
            logger.debug(
                f"Entities mentioned in clarification text: {entities_mentioned}"
            )

            if entities_mentioned:
                logger.info(
                    f"Clarification suggests retrying entities: {entities_mentioned}"
                )
                for entity_id in entities_mentioned:
                    try:
                        self._kb.clear_obstacles_for_entity(entity_id)
                        logger.info(
                            f"Successfully cleared obstacles for entity '{entity_id}'. Adding to retry priority list."
                        )
                        self._entities_to_prioritize_retry.append(entity_id)
                    except KBInteractionError as kbe:
                        logger.error(
                            f"Failed to clear obstacles for '{entity_id}' via KB: {kbe}",
                            exc_info=True,
                        )
                        # Raise PlanningError to signal clarification processing failure
                        raise PlanningError(
                            f"Failed to clear obstacles for '{entity_id}': {kbe.message}",
                            operation_data=kbe.data,
                        ) from kbe
                    except Exception as e:
                        logger.error(
                            f"Unexpected error clearing obstacles for '{entity_id}': {e}",
                            exc_info=True,
                        )
                        # Raise PlanningError for unexpected issues
                        raise PlanningError(
                            f"Unexpected error clearing obstacles for '{entity_id}': {e}"
                        ) from e
            else:
                logger.info(
                    "Clarification mentioned retry, but no specific known entities identified in the text."
                )
        else:
            logger.debug("Clarification did not contain retry keywords.")

        # TODO: Implement more sophisticated parsing or state updates based on clarification if needed.

    def _plan_discovery(self) -> DiscoveryDirectiveContent:
        """Generates parameters for a Discovery Directive."""
        if not self._task_definition:
            op_data = OperationalErrorData(
                component="Planner",
                operation="_plan_discovery",
                details="TaskDefinition not set.",
            )
            raise PlanningError(
                "Cannot plan discovery without TaskDefinition.", operation_data=op_data
            )

        logger.info("Planner: Planning Discovery Directive.")
        inquiry_focus = "Action: Discover initial entities relevant to the task."
        focus_areas = [self._task_definition.task_description]
        if self._task_definition.hints:
            focus_areas.append(f"Hints: {self._task_definition.hints}")
        # Ensure target_entity_id is explicitly None for Discovery
        directive_content: DiscoveryDirectiveContent = (
            inquiry_focus,
            focus_areas,
            None,
        )
        return directive_content

    def _evaluate_candidates(
        self, entity_ids: List[str]
    ) -> Tuple[List[CandidateEvaluation], List[str]]:
        """
        Evaluates a list of candidate entity IDs based on KB data.

        Args:
            entity_ids: A list of entity IDs to evaluate.

        Returns:
            A tuple containing:
                - A list of CandidateEvaluation dictionaries for unblocked candidates.
                - A list of entity IDs for blocked candidates.
        """
        if not self._task_definition:
            op_data = OperationalErrorData(
                component="Planner",
                operation="_evaluate_candidates",
                details="TaskDefinition not set.",
            )
            raise PlanningError(
                "Cannot evaluate candidates without TaskDefinition.",
                operation_data=op_data,
            )

        candidate_evaluations: List[CandidateEvaluation] = []
        blocked_candidates: List[str] = []
        proposal_boosts = defaultdict(float)  # Store boosts derived from proposals
        all_known_entity_ids = set()  # Use set for faster lookups

        # Fetch all known entity IDs once for efficient proposal parsing
        try:
            all_known_entity_ids = set(self._kb.get_all_entity_ids())
            logger.debug(
                f"Fetched {len(all_known_entity_ids)} known entity IDs for proposal parsing."
            )
        except KBInteractionError as kbe:
            logger.error(
                f"Failed to get entity IDs for proposal parsing: {kbe}", exc_info=True
            )
            # Continue without proposal parsing if KB fails, but log error
        except Exception as e:
            logger.error(
                f"Unexpected error getting entity IDs for proposal parsing: {e}",
                exc_info=True,
            )
            # Continue without proposal parsing

        for entity_id in entity_ids:
            logger.debug(f"Planner: Evaluating entity '{entity_id}' for enrichment.")
            profile = None  # Initialize profile to None
            try:
                profile = self._kb.get_entity_profile(entity_id)  # <-- KB Read
            except Exception as e:
                # Log the error but don't raise immediately, treat as profile load failure for this entity
                logger.error(
                    f"Failed to get profile for entity '{entity_id}': {e}",
                    exc_info=True,
                )
                op_data = OperationalErrorData(
                    component="KnowledgeBase",
                    operation="get_entity_profile",
                    details=str(e),
                )
                logger.warning(
                    f"KB Interaction Warning during candidate eval: {op_data.model_dump_json()}",
                )
                # profile remains None

            # Check if ANY profile row for this entity has obstacles
            has_obstacles = False
            if isinstance(profile, list):  # Handle list case (MultiIndex)
                for row_profile in profile:
                    if row_profile and row_profile.get(OBSTACLES_COL):
                        has_obstacles = True
                        logger.warning(
                            f"Planner: Entity '{entity_id}' has known obstacles in row {row_profile.get(self._kb._data_store.index_columns[1], 'N/A')}: {row_profile.get(OBSTACLES_COL, 'Unknown Obstacle')}. Marking as blocked."
                        )
                        break  # Found an obstacle, no need to check other rows
            elif isinstance(profile, dict):  # Handle dict case (Single Index)
                if profile.get(OBSTACLES_COL):
                    has_obstacles = True
                    logger.warning(
                        f"Planner: Entity '{entity_id}' has known obstacles: {profile.get(OBSTACLES_COL, 'Unknown Obstacle')}. Marking as blocked."
                    )

            if has_obstacles:
                blocked_candidates.append(entity_id)
                continue  # Skip further evaluation for blocked candidates

            # --- Calculate Priority Score ---
            priority_score = 0.0  # Use float for potentially finer scoring

            # Extract confidence and proposals, handling list or dict profile
            confidence = None
            proposals = None
            if profile:
                if isinstance(profile, list):
                    if profile:  # Check if list is not empty
                        first_row_profile = profile[0]
                        confidence = first_row_profile.get(CONFIDENCE_COL)
                        proposals = first_row_profile.get(PROPOSALS_COL)
                        # findings = first_row_profile.get(FINDINGS_COL) # Not used yet
                    else:
                        logger.warning(
                            f"Profile for entity '{entity_id}' was an empty list."
                        )
                elif isinstance(profile, dict):
                    confidence = profile.get(CONFIDENCE_COL)
                    proposals = profile.get(PROPOSALS_COL)
                    # findings = profile.get(FINDINGS_COL) # Not used yet
                else:
                    logger.warning(
                        f"Unexpected profile type for entity '{entity_id}': {type(profile)}"
                    )

                # Now use the extracted confidence and proposals variables

                # Boost for low confidence
                if (
                    isinstance(confidence, (int, float))
                    and confidence < CONFIDENCE_THRESHOLD
                ):
                    priority_score += LOW_CONFIDENCE_BOOST
                    logger.debug(
                        f"Candidate {entity_id} low confidence ({confidence:.2f}), boosting priority."
                    )

                # Boost for existing proposals (generic boost + specific entity mention boost)
                if proposals and isinstance(proposals, list):
                    # Apply small generic boost just for having proposals
                    priority_score += PROPOSAL_BOOST
                    logger.debug(
                        f"Candidate {entity_id} has proposals, applying generic boost {PROPOSAL_BOOST}."
                    )
                    # Parse proposals to boost *other* mentioned entities
                    if (
                        all_known_entity_ids
                    ):  # Only parse if we successfully fetched IDs
                        for proposal_item in proposals:
                            if (
                                isinstance(proposal_item, dict)
                                and "proposal" in proposal_item
                                and isinstance(proposal_item["proposal"], str)
                            ):
                                proposal_text = proposal_item["proposal"].lower()
                                for known_id in all_known_entity_ids:
                                    # Check if known_id (lowercase) is in proposal_text and not the current entity
                                    if (
                                        known_id != entity_id
                                        and known_id.lower() in proposal_text
                                    ):
                                        proposal_boosts[
                                            known_id
                                        ] += MENTIONED_IN_PROPOSAL_BOOST
                                        logger.info(
                                            f"Boosting '{known_id}' by {MENTIONED_IN_PROPOSAL_BOOST} due to mention in proposal from '{entity_id}'."
                                        )

                # High boost if explicitly requested for retry via clarification
                if entity_id in self._entities_to_prioritize_retry:
                    priority_score += RETRY_PRIORITY_BOOST
                    logger.info(
                        f"Candidate {entity_id} requested for retry, boosting priority significantly by {RETRY_PRIORITY_BOOST}."
                    )

            else:  # This case now covers both actual None profile and KB read failures
                logger.warning(
                    f"Could not load profile for candidate '{entity_id}'. Assigning default priority."
                )

            # Determine missing attributes now for potential use in selection/focus
            missing_attributes = []
            if profile and self._task_definition.target_columns:
                # Determine profile to check based on type (list or dict)
                profile_to_check = None
                if isinstance(profile, list):
                    if profile:  # Check if list is not empty
                        profile_to_check = profile[0]
                elif isinstance(profile, dict):
                    profile_to_check = profile

                if profile_to_check:
                    missing_attributes = [
                        col
                        for col in self._task_definition.target_columns
                        if pd.isnull(
                            profile_to_check.get(col)
                        )  # Check the determined profile
                    ]
                else:
                    # If profile was None or an empty list, assume all targets are missing
                    missing_attributes = self._task_definition.target_columns[:]

            candidate_evaluations.append(
                {
                    "id": entity_id,
                    "profile": profile,  # Can be None if KB read failed
                    "priority": priority_score,
                    "missing_attributes": missing_attributes,
                }
            )

        # --- Apply Accumulated Proposal Boosts ---
        if proposal_boosts:
            logger.debug(
                f"Applying accumulated proposal boosts: {dict(proposal_boosts)}"
            )
            for evaluation in candidate_evaluations:
                boost = proposal_boosts.get(evaluation["id"], 0.0)
                if boost > 0:
                    original_priority = evaluation["priority"]
                    evaluation["priority"] += boost
                    logger.info(
                        f"Adjusted priority for '{evaluation['id']}' from {original_priority:.2f} to {evaluation['priority']:.2f} based on proposal mentions."
                    )

        return candidate_evaluations, blocked_candidates

    def _select_best_candidate(
        self, evaluations: List[CandidateEvaluation]
    ) -> Optional[CandidateEvaluation]:
        """Selects the best candidate from the evaluated list."""
        if not evaluations:
            return None

        # Sort by priority (descending), then by entity ID (ascending) for deterministic tie-breaking
        evaluations.sort(key=lambda x: (-x["priority"], x["id"]))

        # Simple selection: pick the highest priority
        selected = evaluations[0]
        logger.info(
            f"Planner: Selected entity '{selected['id']}' (Priority: {selected['priority']:.2f}) for next enrichment inquiry."
        )
        return selected

    def _determine_focus_areas(self, candidate: CandidateEvaluation) -> List[str]:
        """Determines the specific attributes to focus on for an enrichment directive."""
        if not self._task_definition:
            op_data = OperationalErrorData(
                component="Planner",
                operation="_determine_focus_areas",
                details="TaskDefinition not set.",
            )
            raise PlanningError(
                "Cannot determine focus areas without TaskDefinition.",
                operation_data=op_data,
            )

        target_columns = self._task_definition.target_columns
        profile = candidate["profile"]
        missing_attributes = candidate["missing_attributes"]  # Already calculated

        if not profile:
            logger.warning(
                f"No profile for selected entity '{candidate['id']}'. Targeting all defined target columns."
            )
            return target_columns[:]  # Return a copy

        focus_areas = missing_attributes[:]  # Start with definitely missing ones

        # --- Adapt Focus Areas Based on Feedback (Example Logic) ---
        # FIX: Access the first element if profile is a non-empty list
        profile_data = None
        if isinstance(profile, list) and profile:
            profile_data = profile[0]
        elif isinstance(profile, dict):
            profile_data = profile

        if profile_data:
            proposals = profile_data.get(PROPOSALS_COL)
            if proposals and isinstance(proposals, list):
                # Example: If proposals suggest specific columns, add them to focus
                for proposal_item in proposals:
                    if (
                        isinstance(proposal_item, dict)
                        and "suggested_focus" in proposal_item
                        and isinstance(proposal_item["suggested_focus"], list)
                    ):
                        for col in proposal_item["suggested_focus"]:
                            if col in target_columns and col not in focus_areas:
                                logger.debug(
                                    f"Adding '{col}' to focus areas based on proposal for '{candidate['id']}'."
                                )
                                focus_areas.append(col)

        # If, after considering missing attributes and proposals, focus_areas is empty,
        # default back to all target columns to ensure progress.
        if not focus_areas:
            logger.warning(
                f"No specific focus areas determined for '{candidate['id']}'. Defaulting to all target columns: {target_columns}"
            )
            return target_columns[:]

        return focus_areas

    def _plan_enrichment(
        self, priority_targets: Optional[List[str]] = None
    ) -> Union[
        EnrichmentDirectiveContent, DirectiveType.CLARIFICATION, None
    ]:  # Use Enum
        """
        Plans the next Enrichment directive or signals completion/clarification needed.

        Prioritizes entities specified in `priority_targets` if provided.
        Otherwise, identifies incomplete entities from the KB.
        Evaluates candidates, selects the best one, determines focus areas,
        and returns the directive content.

        Args:
            priority_targets: A list of entity IDs to prioritize for enrichment.

        Returns:
            - EnrichmentDirectiveContent tuple if a directive can be planned.
            - "CLARIFICATION_NEEDED" if all viable candidates are blocked.
            - None if no incomplete entities are found or no viable candidate selected.
        """
        if not self._task_definition:
            op_data = OperationalErrorData(
                component="Planner",
                operation="_plan_enrichment",
                details="TaskDefinition not set.",
            )
            raise PlanningError(
                "Cannot plan enrichment without TaskDefinition.", operation_data=op_data
            )

        candidate_ids: List[str] = []
        if priority_targets:
            logger.info(
                f"Planner: Planning enrichment with priority targets: {priority_targets}"
            )
            candidate_ids = priority_targets
        else:
            # If no priority targets, find all incomplete entities based on granularity
            try:
                candidate_ids = self._kb.find_incomplete_entities(
                    self._task_definition.target_columns
                )  # <-- KB Read
                logger.info(
                    f"Planner: Found {len(candidate_ids)} incomplete entities: {candidate_ids}"
                )
            except KBInteractionError as kbe:
                logger.error(
                    f"Failed to find incomplete entities: {kbe}", exc_info=True
                )
                # Treat as no candidates found if KB fails
                candidate_ids = []
            except Exception as e:
                logger.error(
                    f"Unexpected error finding incomplete entities: {e}", exc_info=True
                )
                candidate_ids = []

        if not candidate_ids:
            logger.info("Planner: No incomplete entities found for enrichment.")
            return None  # Signal completion for this phase

        # --- Evaluate Candidates ---
        try:
            viable_candidates, blocked_candidates = self._evaluate_candidates(
                candidate_ids
            )
        except PlanningError:
            raise  # Propagate planning errors during evaluation
        except Exception as e:
            logger.error(
                f"Unexpected error during candidate evaluation: {e}", exc_info=True
            )
            # Treat as no viable candidates if evaluation fails unexpectedly
            viable_candidates = []
            blocked_candidates = candidate_ids  # Assume all are potentially blocked

        logger.info(
            f"Planner: Evaluated candidates. Viable: {len(viable_candidates)}, Blocked: {len(blocked_candidates)} ({blocked_candidates})"
        )

        # --- Select Best Candidate ---
        selected_candidate = self._select_best_candidate(viable_candidates)

        if not selected_candidate:
            if not viable_candidates and blocked_candidates:
                logger.warning(
                    "Planner: All remaining incomplete entities have known obstacles. Signaling for user clarification."
                )
                return DirectiveType.CLARIFICATION  # Use Enum
            elif not viable_candidates and not blocked_candidates:
                # This case might happen if profile loading failed for all candidates
                logger.error(
                    "Planner: No viable candidates found (all failed profile load or other eval error?). Signaling completion to avoid loop."
                )
                return None
            else:  # Should not happen if _select_best_candidate works correctly
                logger.error(
                    "Planner: Candidate selection failed unexpectedly after filtering. Signaling completion."
                )
                return None

        # --- Determine Focus & Formulate Directive ---
        focus_areas = self._determine_focus_areas(selected_candidate)
        if not focus_areas:
            logger.warning(
                f"Entity '{selected_candidate['id']}' was selected, but no focus areas determined (might be complete?). Signaling completion to avoid loop."
            )
            # If an entity is selected but has no missing target columns, treat as complete for now.
            return None

        target_entity_id = selected_candidate["id"]
        inquiry_focus = f"Action: Enrich entity '{target_entity_id}'"
        logger.info(
            f"Planner: Enrichment directive for '{target_entity_id}': Focus on {focus_areas}"
        )
        # Ensure target_entity_id is not None for Enrichment
        directive_content: EnrichmentDirectiveContent = (
            inquiry_focus,
            focus_areas,
            target_entity_id,  # This is now guaranteed to be a string ID
        )
        # Clear the retry list now that a selection has been made based on it
        self._entities_to_prioritize_retry = []
        return directive_content  # Return the tuple directly

    def determine_next_directive(self) -> PlannerSignal:
        """
        Determines the next directive signal based on the current Knowledge Base state.

        Handles strategic review triggers (obstacles, confidence, stagnation) and
        plans Discovery or Enrichment directives accordingly.

        Returns:
            A PlannerSignal indicating the next action for the MCP Interface.
        Raises:
            PlanningError: If prerequisites are missing or logic fails.
            KBInteractionError: If KB interaction fails.
        """
        if not self._kb.is_ready or not self._task_definition:
            logger.error(
                "Precondition failed for planning: KnowledgeBase not ready or TaskDefinition not set."
            )
            op_data = OperationalErrorData(
                component="Planner",
                operation="determine_next_directive",
                details="KB not ready or TaskDefinition missing.",
            )
            raise PlanningError(
                "Cannot plan next directive due to missing prerequisites.",
                operation_data=op_data,
            )

        # Note: Report processing is now synchronous, so KB reads below reflect the latest submitted data.
        try:
            kb_summary = self._kb.get_knowledge_summary()  # <-- KB Read
        except Exception as e:
            logger.error(f"Failed to get KB summary for planning: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation="get_knowledge_summary",
                details=str(e),
            )
            raise KBInteractionError(
                f"Failed to get KB summary: {e}", operation_data=op_data
            ) from e

        logger.debug(f"Planner: KB Summary: {kb_summary}")

        # --- Initial Discovery ---
        if kb_summary.get("entity_count", 0) == 0:
            # If KB is empty, always start with discovery
            logger.info("Planner: KB empty. Planning discovery.")
            # Reset processed base entities when starting discovery
            self._processed_base_entities_enrichment = set()
            discovery_content = self._plan_discovery()
            return (DirectiveType.DISCOVERY, discovery_content)

        # --- Check for Obstacles Blocking ALL Paths ---
        # TODO: Implement a more robust check if needed

        # --- Enrichment / Review / Completion Logic ---
        # Reset processed base entities set if it's empty (start of a new cycle after discovery/review)
        if not self._processed_base_entities_enrichment:
            logger.info(
                "Planner: Resetting processed base entities set for new enrichment cycle."
            )
            self._processed_base_entities_enrichment = set()

        # 1. Prioritize Granular Incomplete Entities
        try:
            granular_incomplete_entities = self._kb.find_incomplete_entities(
                self._task_definition.target_columns
            )  # <-- KB Read
        except Exception as e:
            logger.error(f"Failed to get incomplete entities: {e}", exc_info=True)
            return ("NEEDS_STRATEGIC_REVIEW", f"Failed to get incomplete entities: {e}")

        if granular_incomplete_entities:
            logger.info(
                f"Planner: Found {len(granular_incomplete_entities)} granular incomplete entities. Planning enrichment."
            )
            enrichment_signal = self._plan_enrichment(
                priority_targets=granular_incomplete_entities
            )
            if enrichment_signal:
                # If enrichment is planned for a granular entity, return it
                if (
                    isinstance(enrichment_signal, tuple)
                    and enrichment_signal[0] == DirectiveType.ENRICHMENT
                ):
                    return (DirectiveType.ENRICHMENT, enrichment_signal[1])
                elif enrichment_signal == "CLARIFICATION_NEEDED":
                    return "CLARIFICATION_NEEDED"
                else:  # Should not happen
                    logger.error(
                        f"Planner: _plan_enrichment returned unexpected signal for granular: {enrichment_signal}"
                    )
                    return (
                        "NEEDS_STRATEGIC_REVIEW",
                        "Internal planner error: Unexpected granular enrichment signal.",
                    )
            else:
                logger.warning(
                    "Planner: Found granular incomplete entities, but _plan_enrichment returned no directive. Might be blocked or error."
                )
                # Fall through to check base entities / stagnation etc.

        # 2. If No Granular Incomplete, Process Next Unprocessed Base Entity
        if not granular_incomplete_entities:
            logger.info(
                "Planner: No granular incomplete entities found. Checking for next unprocessed base entity."
            )
            try:
                all_base_entity_ids = set(self._kb.get_all_entity_ids())  # <-- KB Read
                # Sort for deterministic processing order
                sorted_base_ids = sorted(list(all_base_entity_ids))

                next_unprocessed_base_entity = next(
                    (
                        eid
                        for eid in sorted_base_ids
                        if eid not in self._processed_base_entities_enrichment
                    ),
                    None,
                )

                if next_unprocessed_base_entity:
                    logger.info(
                        f"Planner: Planning enrichment for next base entity: '{next_unprocessed_base_entity}'"
                    )
                    self._processed_base_entities_enrichment.add(
                        next_unprocessed_base_entity
                    )
                    enrichment_signal = self._plan_enrichment(
                        priority_targets=[next_unprocessed_base_entity]
                    )
                    if enrichment_signal:
                        if (
                            isinstance(enrichment_signal, tuple)
                            and enrichment_signal[0] == DirectiveType.ENRICHMENT
                        ):
                            return (DirectiveType.ENRICHMENT, enrichment_signal[1])
                        elif enrichment_signal == "CLARIFICATION_NEEDED":
                            # If the chosen base entity needs clarification, signal it
                            return "CLARIFICATION_NEEDED"
                        else:  # Should not happen
                            logger.error(
                                f"Planner: _plan_enrichment returned unexpected signal for base: {enrichment_signal}"
                            )
                            return (
                                "NEEDS_STRATEGIC_REVIEW",
                                "Internal planner error: Unexpected base enrichment signal.",
                            )
                    else:
                        logger.warning(
                            f"Planner: Tried to plan enrichment for base entity '{next_unprocessed_base_entity}', but received no directive. Might be blocked or error."
                        )
                        # Fall through to stagnation checks etc.
                else:
                    # --- All Base Entities Processed ---
                    logger.info(
                        "Planner: All known base entities have been processed in this enrichment cycle."
                    )

                    # 3. Stagnation & Confidence Checks (Only if all base entities processed)
                    logger.debug(
                        "Planner: Checking stagnation and confidence before strategic review."
                    )
                    try:
                        # Check for stagnation using granular incomplete set
                        # Re-fetch granular incomplete set for the check
                        current_incomplete_set = set(
                            self._kb.find_incomplete_entities(
                                self._task_definition.target_columns
                            )
                        )
                        if (
                            self._last_incomplete_set is not None
                            and current_incomplete_set == self._last_incomplete_set
                        ):
                            self._stagnation_counter += 1
                            logger.warning(
                                f"Planner: Incomplete set unchanged. Stagnation counter: {self._stagnation_counter}"
                            )
                        else:
                            logger.debug(
                                "Planner: Incomplete set changed or first check. Resetting stagnation counter."
                            )
                            self._stagnation_counter = 0
                            self._last_incomplete_set = current_incomplete_set

                        if (
                            self._stagnation_counter
                            >= self._stagnation_cycles_threshold
                        ):
                            logger.error(
                                f"Planner: Stagnation detected ({self._stagnation_counter} cycles). Signaling strategic review."
                            )
                            # Reset state for next cycle after review
                            self._processed_base_entities_enrichment = set()
                            self._stagnation_counter = 0
                            self._last_incomplete_set = None
                            return (
                                "NEEDS_STRATEGIC_REVIEW",
                                f"Stagnation detected after {self._stagnation_counter} cycles with no progress on incomplete entities.",
                            )

                        # Check overall confidence
                        avg_confidence = (
                            self._kb.get_average_confidence()
                        )  # <-- KB Read
                        if (
                            avg_confidence is not None
                            and avg_confidence < self._confidence_threshold
                        ):
                            logger.warning(
                                f"Planner: Average KB confidence ({avg_confidence:.2f}) is below threshold ({self._confidence_threshold}). Signaling strategic review."
                            )
                            # Reset state for next cycle after review
                            self._processed_base_entities_enrichment = set()
                            return (
                                "NEEDS_STRATEGIC_REVIEW",
                                f"Average KB confidence ({avg_confidence:.2f}) is below threshold ({self._confidence_threshold}).",
                            )

                    except KBInteractionError as kbe:
                        logger.error(
                            f"Planner: KB error during stagnation/confidence check: {kbe}",
                            exc_info=True,
                        )
                        return (
                            "NEEDS_STRATEGIC_REVIEW",
                            f"KB error during planner checks: {kbe.message}",
                        )
                    except Exception as e:
                        logger.error(
                            f"Planner: Unexpected error during stagnation/confidence check: {e}",
                            exc_info=True,
                        )
                        return (
                            "NEEDS_STRATEGIC_REVIEW",
                            f"Unexpected error during planner checks: {e}",
                        )

                    # 4. Trigger Strategic Review (If all base entities processed & no other issues)
                    logger.info(
                        "Planner: Initial enrichment cycle complete for all known entities. No stagnation or low confidence detected. Signaling strategic review."
                    )
                    # Reset processed set before signaling review, so next cycle starts fresh
                    self._processed_base_entities_enrichment = set()
                    return (
                        "NEEDS_STRATEGIC_REVIEW",
                        "Initial enrichment phase complete. Review results and provide guidance for the next phase (e.g., 'continue discovery', 'refine targets', 'complete research').",
                    )

            except KBInteractionError as kbe:
                logger.error(
                    f"Planner: KB error while getting base entity IDs: {kbe}",
                    exc_info=True,
                )
                return (
                    "NEEDS_STRATEGIC_REVIEW",
                    f"KB error during planner checks: {kbe.message}",
                )
            except Exception as e:
                logger.error(
                    f"Planner: Unexpected error while getting base entity IDs: {e}",
                    exc_info=True,
                )
                return (
                    "NEEDS_STRATEGIC_REVIEW",
                    f"Unexpected error during planner checks: {e}",
                )

        # Fallback if somehow no directive was issued (should ideally not be reached with new logic)
        logger.error(
            "Planner: Reached end of planning logic without issuing a directive or review signal. This indicates a potential logic error."
        )
        return (
            "NEEDS_STRATEGIC_REVIEW",
            "Planner reached unexpected state. Needs review.",
        )

    # Removed check_completion method as completion is now determined within determine_next_directive
    # based on planner outcome or client assessment.
