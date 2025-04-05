import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

# Import custom exceptions and error models
from ..core.exceptions import KBInteractionError, PlanningError
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
    Tuple[Literal["ENRICH"], EnrichmentDirectiveContent],
    Tuple[Literal["DISCOVERY"], DiscoveryDirectiveContent],
    Tuple[Literal["NEEDS_STRATEGIC_REVIEW"], str],  # Includes the reason
    Literal["CLARIFICATION_NEEDED"],  # If obstacles block all paths
    None,  # Indicates research completion
]

# Constants for prioritization logic

# Strategic review thresholds are now instance attributes (_obstacle_threshold, _confidence_threshold)

CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_BOOST = 2.0  # Use float
PROPOSAL_BOOST = 1.0  # Use float
# RECENCY_PENALTY_FACTOR = 0.1 # Example: Penalize recently attempted entities slightly (Not implemented yet)


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
        """Processes user clarification to potentially influence future planning."""
        if clarification:
            logger.info(
                f"Planner received user clarification: '{clarification[:100]}...'"
            )
            # TODO: Implement logic to parse clarification and update planner state/strategy
            # For now, just logging it is sufficient to fix the AttributeError
            pass
        else:
            logger.info("Planner received empty clarification.")

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

            if profile and profile.get(OBSTACLES_COL):
                logger.warning(
                    f"Planner: Entity '{entity_id}' has known obstacles: {profile[OBSTACLES_COL]}. Marking as blocked."
                )
                blocked_candidates.append(entity_id)
                continue  # Skip further evaluation for blocked candidates

            # --- Calculate Priority Score ---
            priority_score = 0.0  # Use float for potentially finer scoring

            if profile:
                confidence = profile.get(CONFIDENCE_COL)
                proposals = profile.get(PROPOSALS_COL)
                # findings = profile.get(FINDINGS_COL) # Not used yet

                # Boost for low confidence
                if (
                    isinstance(confidence, (int, float))
                    and confidence < CONFIDENCE_THRESHOLD
                ):
                    priority_score += LOW_CONFIDENCE_BOOST
                    logger.debug(
                        f"Candidate {entity_id} low confidence ({confidence:.2f}), boosting priority."
                    )

                # Boost for existing proposals
                if proposals:
                    priority_score += PROPOSAL_BOOST
                    logger.debug(
                        f"Candidate {entity_id} has proposals, boosting priority."
                    )

            else:  # This case now covers both actual None profile and KB read failures
                logger.warning(
                    f"Could not load profile for candidate '{entity_id}'. Assigning default priority."
                )

            # Determine missing attributes now for potential use in selection/focus
            missing_attributes = []
            if profile and self._task_definition.target_columns:
                missing_attributes = [
                    col
                    for col in self._task_definition.target_columns
                    if pd.isnull(profile.get(col))
                ]

            candidate_evaluations.append(
                {
                    "id": entity_id,
                    "profile": profile,  # Can be None if KB read failed
                    "priority": priority_score,
                    "missing_attributes": missing_attributes,
                }
            )

        return candidate_evaluations, blocked_candidates

    def _select_best_candidate(
        self, evaluations: List[CandidateEvaluation]
    ) -> Optional[CandidateEvaluation]:
        """Selects the best candidate from the evaluated list."""
        if not evaluations:
            return None

        # Sort by priority (descending)
        evaluations.sort(key=lambda x: x["priority"], reverse=True)

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
        proposals = profile.get(PROPOSALS_COL)
        if proposals and isinstance(proposals, list):
            logger.debug(
                f"Entity {candidate['id']} has proposals. Adapting focus areas."
            )
            try:
                # Example: If a proposal mentions a target column, ensure it's in focus
                for prop_item in proposals:
                    if (
                        isinstance(prop_item, dict)
                        and "proposal" in prop_item
                        and isinstance(prop_item["proposal"], str)
                    ):
                        proposal_text = prop_item["proposal"].lower()
                        for col in target_columns:
                            if col.lower() in proposal_text and col not in focus_areas:
                                logger.info(
                                    f"Adapting focus for {candidate['id']}: Adding '{col}' based on proposal mentioning it."
                                )
                                focus_areas.append(col)
            except Exception as e:
                logger.error(
                    f"Error processing proposals for focus adaptation: {e}",
                    exc_info=True,
                )

        # FUTURE: Similar logic could be added for findings

        # Ensure focus_areas only contains valid columns from the task definition
        valid_focus_areas = [
            area for area in focus_areas if area in self._task_definition.columns
        ]

        if not valid_focus_areas:
            logger.warning(
                f"Focus area adaptation resulted in empty list for entity '{candidate['id']}'. Falling back to initially missing attributes or target columns."
            )
            return missing_attributes[:] if missing_attributes else target_columns[:]

        return list(set(valid_focus_areas))  # Return unique list

    def _plan_enrichment(
        self, priority_targets: Optional[List[str]] = None
    ) -> Optional[Union[EnrichmentDirectiveContent, Literal["CLARIFICATION_NEEDED"]]]:
        """
        Plans the next enrichment step or signals completion/clarification.

        Args:
            priority_targets: Optional list of entity IDs to prioritize.

        Returns:
            - EnrichmentDirectiveContent tuple if a target is found.
            - "CLARIFICATION_NEEDED" if all incomplete entities are blocked.
            - None if no incomplete entities are found.
        Raises:
            PlanningError: If TaskDefinition is not set.
            KBInteractionError: If KB interaction fails.
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

        logger.debug("Planner: KB not empty, proceeding with enrichment planning.")
        target_columns = self._task_definition.target_columns
        if not target_columns:
            logger.warning(
                "No target columns defined. Cannot plan enrichment. Considering research complete."
            )
            return None  # Signal completion

        # Note: Report processing is now synchronous. KB reads reflect latest submitted data.
        try:
            incomplete_entities = self._kb.find_incomplete_entities(
                target_columns
            )  # <-- KB Read
        except Exception as e:
            logger.error(f"Failed to find incomplete entities: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation="find_incomplete_entities",
                details=str(e),
            )
            raise KBInteractionError(
                f"Failed to find incomplete entities: {e}", operation_data=op_data
            ) from e
        logger.debug(f"Planner: Found incomplete entities: {incomplete_entities}")

        if not incomplete_entities:
            logger.info(
                "Planner: All known entities appear complete based on target attributes. Research complete."
            )
            logger.info("Planner: No incomplete entities found. Signaling completion.")
            return None  # Signal completion

        # --- Evaluate Candidates ---
        candidate_evaluations, blocked_candidates = self._evaluate_candidates(
            incomplete_entities
        )

        # --- Handle Blockers / Select Candidate (with Priority Targets) ---
        viable_candidates = candidate_evaluations  # Start with all unblocked

        # If priority targets are given, try to select from them first
        if priority_targets:
            priority_evals = [
                cand for cand in viable_candidates if cand["id"] in priority_targets
            ]
            if priority_evals:
                logger.debug(
                    f"Prioritizing selection from {len(priority_evals)} client-specified targets."
                )
                selected_candidate = self._select_best_candidate(priority_evals)
                if selected_candidate:
                    logger.info(f"Selected priority target: {selected_candidate['id']}")
                else:  # Should not happen if priority_evals is not empty, but safety check
                    logger.warning(
                        "Priority target selection failed unexpectedly, falling back."
                    )
                    selected_candidate = self._select_best_candidate(
                        viable_candidates
                    )  # Fallback
            else:
                logger.debug(
                    "No valid/unblocked priority targets found, selecting from all viable candidates."
                )
                selected_candidate = self._select_best_candidate(viable_candidates)
        else:
            # No priority targets, select from all viable candidates
            selected_candidate = self._select_best_candidate(viable_candidates)

        # Handle cases where no candidate could be selected or all are blocked
        if not selected_candidate:
            if not viable_candidates and blocked_candidates:
                logger.warning(
                    "Planner: All remaining incomplete entities have known obstacles. Signaling for user clarification."
                )
                return "CLARIFICATION_NEEDED"
            elif not viable_candidates and not blocked_candidates:
                logger.error(
                    "Planner: No viable candidates found (all failed profile load?). Signaling completion."
                )
                return None  # Avoid potential loops
            else:  # Should not happen if _select_best_candidate works correctly
                logger.error(
                    "Planner: Candidate selection failed unexpectedly after filtering. Signaling completion."
                )
                return None

        # --- Determine Focus & Formulate Directive ---
        focus_areas = self._determine_focus_areas(selected_candidate)
        if not focus_areas:
            logger.warning(
                f"Entity '{selected_candidate['id']}' was selected, but no focus areas determined. Signaling completion to avoid loop."
            )
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
        return directive_content  # Return the tuple directly

    def determine_next_directive(
        self, client_assessment: Optional[Dict[str, Any]] = None
    ) -> PlannerSignal:
        """
        Determines the next directive signal based on KB state and optional client assessment.

        Args:
            client_assessment: Optional dictionary representing the structured assessment
                               provided by the client in the last report.

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

        # --- Priority 0: Check for Strategic Review Triggers ---
        entity_count = kb_summary.get("entity_count", 0)
        if entity_count > 0:  # Only check triggers if there are entities
            try:
                # Check Obstacle Threshold
                entities_with_obstacles = (
                    self._kb.get_entities_with_active_obstacles()
                )  # <-- KB Read
                obstacle_ratio = len(entities_with_obstacles) / entity_count
                if obstacle_ratio >= self._obstacle_threshold:
                    logger.warning(
                        f"Planner: Obstacle threshold met ({obstacle_ratio:.2f} >= {self._obstacle_threshold}). Triggering Strategic Review."
                    )
                    return ("NEEDS_STRATEGIC_REVIEW", "critical_obstacles")

                # Check Confidence Threshold
                avg_confidence = self._kb.get_average_confidence()  # <-- KB Read
                if (
                    avg_confidence is not None
                    and avg_confidence < self._confidence_threshold
                ):
                    logger.warning(
                        f"Planner: Average confidence threshold met ({avg_confidence:.2f} < {self._confidence_threshold}). Triggering Strategic Review."
                    )
                    return ("NEEDS_STRATEGIC_REVIEW", "low_confidence")

                # Check Stagnation
                current_incomplete_set = set(
                    self._kb.find_incomplete_entities(
                        self._task_definition.target_columns
                    )
                )  # <-- KB Read
                if (
                    self._last_incomplete_set is not None
                    and current_incomplete_set == self._last_incomplete_set
                ):
                    self._stagnation_counter += 1
                    logger.debug(
                        f"Planner: Incomplete set unchanged. Stagnation counter: {self._stagnation_counter}"
                    )
                else:
                    logger.debug(
                        "Planner: Incomplete set changed or first check. Resetting stagnation counter."
                    )
                    self._stagnation_counter = 0
                    self._last_incomplete_set = current_incomplete_set

                if self._stagnation_counter >= self._stagnation_cycles_threshold:
                    logger.warning(
                        f"Planner: Stagnation threshold met ({self._stagnation_counter} >= {self._stagnation_cycles_threshold}). Triggering Strategic Review."
                    )
                    # Reset counter after triggering review to avoid immediate re-trigger
                    self._stagnation_counter = 0
                    self._last_incomplete_set = None  # Reset last set as well
                    return ("NEEDS_STRATEGIC_REVIEW", "stagnation")

            except KBInteractionError as kbe:
                logger.error(
                    f"KB interaction failed during strategic review checks: {kbe}",
                    exc_info=True,
                )
                # Decide how to handle - proceed with standard planning or raise?
                # Proceeding might be safer to avoid getting stuck.
            except Exception as e:
                logger.error(
                    f"Unexpected error during strategic review checks: {e}",
                    exc_info=True,
                )
                # Proceed with standard planning

        # --- Priority 1: Client Overrides (if assessment provided) ---
        if client_assessment:
            logger.debug(f"Processing with client assessment: {client_assessment}")
            if client_assessment.get("goal_achieved") is True:
                logger.info(
                    "Planner: Client assessed goal_achieved=True. Signaling completion."
                )
                return None
            if client_assessment.get("request_user_clarification") is True:
                logger.warning(
                    "Planner: Client requested user clarification. Signaling."
                )
                return "CLARIFICATION_NEEDED"
            # Note: We'll use discovery_needed and enrichment_needed flags below

        # --- Priority 2: Server Sanity Checks (Example: Critical Obstacles) ---
        # TODO: Implement more robust obstacle checking if needed, e.g., checking specific critical entities.
        # For now, rely on _plan_enrichment returning "CLARIFICATION_NEEDED" if all incomplete are blocked.

        # --- Priority 3: Guided Action (Based on Client Assessment or Default) ---
        discovery_requested = (
            client_assessment.get("discovery_needed") if client_assessment else None
        )
        enrichment_requested = (
            client_assessment.get("enrichment_needed") if client_assessment else None
        )
        priority_targets = (
            client_assessment.get("prioritized_enrichment_targets")
            if client_assessment
            else None
        )

        if discovery_requested is True:
            logger.info("Planner: Client requested discovery. Planning discovery.")
            discovery_params = self._plan_discovery()
            return ("DISCOVERY", discovery_params)

        if enrichment_requested is True:
            logger.info(
                f"Planner: Client requested enrichment (Priority Targets: {priority_targets}). Planning enrichment."
            )
            enrichment_outcome = self._plan_enrichment(
                priority_targets=priority_targets
            )
            if isinstance(enrichment_outcome, tuple):  # Enrichment directive content
                return ("ENRICH", enrichment_outcome)
            elif enrichment_outcome == "CLARIFICATION_NEEDED":
                logger.warning(
                    "Planner: Client requested enrichment, but all targets are blocked. Signaling clarification."
                )
                return "CLARIFICATION_NEEDED"
            else:  # Enrichment outcome was None (no targets found even with priority)
                logger.warning(
                    "Planner: Client requested enrichment, but no valid targets found. Requesting strategic review."
                )
                # This case indicates a potential mismatch or issue, trigger review.
                return ("NEEDS_STRATEGIC_REVIEW", "client_requested_enrichment_failed")

        # --- Priority 4: Default Algorithmic Action (No specific client guidance for phase) ---
        logger.debug(
            "Planner: No specific client phase guidance, proceeding with default logic."
        )
        if kb_summary.get("entity_count", 0) == 0:
            logger.debug("Planner: Defaulting to discovery (KB empty).")
            discovery_params = self._plan_discovery()
            return ("DISCOVERY", discovery_params)
        else:
            logger.debug("Planner: Defaulting to enrichment planning.")
            enrichment_outcome = self._plan_enrichment()  # No priority targets here

            if isinstance(enrichment_outcome, tuple):  # Enrichment directive content
                return ("ENRICH", enrichment_outcome)
            elif enrichment_outcome == "CLARIFICATION_NEEDED":
                logger.warning(
                    "Planner: Default enrichment found only blocked targets. Requesting strategic review."
                )
                # Instead of direct clarification, let client review the blocked state
                return ("NEEDS_STRATEGIC_REVIEW", "critical_obstacles")
            else:  # enrichment_outcome is None (no incomplete targets found)
                logger.info(
                    "Planner: Default enrichment found no targets. Requesting strategic review."
                )
                return ("NEEDS_STRATEGIC_REVIEW", "enrichment_complete")

        # --- Safety Net ---
        logger.error(
            "Planner: determine_next_directive reached end without returning a signal. Raising PlanningError."
        )
        op_data = OperationalErrorData(
            component="Planner",
            operation="determine_next_directive",
            details="Logical flow error: No planning signal determined.",
        )
        raise PlanningError(
            "Planner failed to determine the next directive.", operation_data=op_data
        )

    # Removed check_completion method as completion is now determined within determine_next_directive
    # based on planner outcome or client assessment.
