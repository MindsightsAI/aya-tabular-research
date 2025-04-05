import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

# Import custom exceptions and error models
from ..core.exceptions import KBInteractionError
from ..core.models.error_models import OperationalErrorData

# Import models needed for type hints and serialization
from ..core.models.research_models import (
    IdentifiedObstacle,
    ProposedNextStep,
    SynthesizedFinding,
    TaskDefinition,
)
from .data_store import DataStore

# Import internal column constants
from .tabular_store import (
    CONFIDENCE_COL,
    FINDINGS_COL,
    NARRATIVE_COL,
    OBSTACLES_COL,
    PROPOSALS_COL,
)

logger = logging.getLogger(__name__)


class KnowledgeBase(ABC):
    """
    Abstract Base Class defining a higher-level interface for accessing
    and managing stored knowledge.
    """

    @abstractmethod
    def initialize(self, task_definition: TaskDefinition):
        """Initializes the knowledge base."""
        pass

    @abstractmethod
    def add_entities(self, entities: List[Dict[str, Any]]):
        """Adds multiple new entities."""
        pass

    @abstractmethod
    def update_entity_attributes(self, entity_id: str, attributes: Dict[str, Any]):
        """Updates attributes for an existing entity."""
        pass

    @abstractmethod
    def batch_update_entities(self, entities: List[Dict[str, Any]]):
        """Adds or updates multiple entities efficiently."""
        pass

    # --- Methods for Richer Data (Phase 3+) ---
    @abstractmethod
    def store_findings(self, entity_id: str, findings: List[SynthesizedFinding]):
        """Stores synthesized findings associated with an entity."""
        pass

    @abstractmethod
    def store_obstacles(self, entity_id: str, obstacles: List[IdentifiedObstacle]):
        """Stores identified obstacles associated with an entity."""
        pass

    @abstractmethod
    def store_proposals(self, entity_id: str, proposals: List[ProposedNextStep]):
        """Stores proposed next steps associated with an entity."""
        pass

    @abstractmethod
    def store_confidence(self, entity_id: str, confidence: Optional[float]):
        """Stores the overall confidence score for a report cycle related to an entity."""
        pass

    @abstractmethod
    def store_narrative(self, entity_id: str, narrative: Optional[str]):
        """Stores the summary narrative for a report cycle related to an entity."""
        pass

    @abstractmethod
    def get_obstacle_summary(self) -> List[Dict[str, Any]]:
        """Retrieves a summary of entities with active obstacles."""
        pass

    @abstractmethod
    def get_findings(self, entity_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves stored findings for a specific entity."""
        pass

    @abstractmethod
    def get_obstacles(self, entity_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves stored obstacles for a specific entity."""
        pass

    @abstractmethod
    def get_proposals(self, entity_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves stored proposals for a specific entity."""
        pass

    @abstractmethod
    def get_confidence(self, entity_id: str) -> Optional[float]:
        """Retrieves the stored confidence score for a specific entity."""
        pass

    @abstractmethod
    def get_narrative(self, entity_id: str) -> Optional[str]:
        """Retrieves the stored narrative for a specific entity."""
        pass

    # --- End Methods for Richer Data ---

    @abstractmethod
    def get_entities_with_active_obstacles(self) -> List[str]:
        """Retrieves IDs of entities that have non-empty obstacle lists."""
        pass

    @abstractmethod
    def get_average_confidence(self) -> Optional[float]:
        """Calculates the average confidence score across entities with scores."""
        pass

    @abstractmethod
    def get_low_confidence_entities(self, threshold: float) -> List[str]:
        """Retrieves IDs of entities with confidence score below a threshold."""
        pass

    @abstractmethod
    def get_entity_profile(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a profile for a specific entity."""
        pass

    @abstractmethod
    def get_full_dataset(self) -> pd.DataFrame:
        """Retrieves the entire structured dataset."""
        pass

    @abstractmethod
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Provides a summary of the current knowledge state."""
        pass

    @abstractmethod
    def find_incomplete_entities(
        self, required_attributes: Optional[List[str]] = None
    ) -> List[str]:
        """Identifies entities missing required user-defined attributes."""
        pass

    @abstractmethod
    def reset_knowledge(self):
        """Resets the knowledge base to an empty state."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the knowledge base is initialized."""
        pass


# --- Concrete Implementation ---


class TabularKnowledgeBase(KnowledgeBase):
    """
    Concrete implementation of KnowledgeBase that uses a TabularStore
    for structured data. This version operates statelessly (in-memory per instance).
    Includes methods for storing richer report data (Phase 3).
    Raises KBInteractionError on failures interacting with the underlying DataStore.
    """

    def __init__(self, data_store: DataStore):
        if not isinstance(data_store, DataStore):
            raise TypeError("data_store must be an instance of DataStore")
        self._data_store = data_store
        self._task_definition: Optional[TaskDefinition] = None
        logger.info("TabularKnowledgeBase initialized.")

    @property
    def is_ready(self) -> bool:
        # Check underlying store's readiness
        try:
            return self._data_store.is_initialized
        except Exception as e:
            logger.error(f"Error checking DataStore readiness: {e}", exc_info=True)
            return False  # Assume not ready if check fails

    def initialize(self, task_definition: TaskDefinition):
        logger.info("Initializing Knowledge Base...")
        self._task_definition = task_definition
        try:
            self._data_store.initialize_storage(task_definition)
            logger.info("Knowledge Base initialized successfully.")
        except Exception as e:
            logger.error(
                f"Failed to initialize Knowledge Base DataStore: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="initialize_storage", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to initialize KB: {e}", operation_data=op_data
            ) from e

    def _raise_if_not_ready(self, operation: str):
        """Helper to raise KBInteractionError if KB is not initialized."""
        if not self.is_ready:
            msg = f"KnowledgeBase (TabularStore) is not initialized. Cannot perform operation: {operation}"
            logger.error(msg)
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation=operation,
                details="KB not initialized.",
            )
            raise KBInteractionError(msg, operation_data=op_data)

    def update_entity_attributes(self, entity_id: str, attributes: Dict[str, Any]):
        """Updates entity attributes in the DataStore."""
        self._raise_if_not_ready("update_entity_attributes")
        try:
            id_col = self._data_store.identifier_column
            if id_col and id_col not in attributes:
                attributes[id_col] = entity_id
            elif id_col and attributes.get(id_col) != entity_id:
                logger.warning(
                    f"Identifier mismatch in update_entity_attributes for {entity_id}. Using provided entity_id."
                )
                attributes[id_col] = entity_id

            self._data_store.add_or_update_entity(entity_id, attributes)
        except Exception as e:
            logger.error(
                f"Failed to update entity '{entity_id}' in KB via DataStore: {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="add_or_update_entity", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to update entity '{entity_id}': {e}", operation_data=op_data
            ) from e

    def add_entities(self, entities: List[Dict[str, Any]]):
        """Adds multiple entities to the DataStore, typically used for seeding."""
        self._raise_if_not_ready("add_entities")
        if not entities:
            logger.debug("add_entities called with empty list, no action taken.")
            return
        try:
            self._data_store.add_entities(entities)
        except Exception as e:
            logger.error(
                f"Failed to add entities to KB via DataStore: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="add_entities", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to add entities: {e}", operation_data=op_data
            ) from e

    def batch_update_entities(self, entities: List[Dict[str, Any]]):
        """Adds or updates multiple entities using the underlying DataStore's batch method."""
        self._raise_if_not_ready("batch_update_entities")
        if not entities:
            logger.debug(
                "batch_update_entities called with empty list, no action taken."
            )
            return
        try:
            self._data_store.batch_add_or_update(entities)
        except Exception as e:
            logger.error(
                f"Failed during batch update in KB via DataStore: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="batch_add_or_update", details=str(e)
            )
            raise KBInteractionError(
                f"Failed during batch update: {e}", operation_data=op_data
            ) from e

    # --- Methods for Storing Richer Data (Phase 3+) ---

    def _store_attribute(self, entity_id: str, column_name: str, data: Any):
        """Helper to store data in an internal column via update_entity_attributes."""
        # _raise_if_not_ready is implicitly called by update_entity_attributes
        try:
            # Pass data directly; TabularStore now stores Python objects
            self.update_entity_attributes(entity_id, {column_name: data})
            logger.debug(f"Stored {column_name} for entity '{entity_id}'.")
        except KBInteractionError:  # Catch error from update_entity_attributes
            logger.error(
                f"Failed to store {column_name} for entity '{entity_id}' due to underlying update failure.",
                exc_info=False,
            )  # Don't log stack trace twice
            raise  # Re-raise the specific error
        except Exception as e:  # Catch unexpected errors in this helper itself
            logger.error(
                f"Unexpected error storing {column_name} for entity '{entity_id}': {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation="_store_attribute",
                details=f"Storing {column_name}: {e}",
            )
            raise KBInteractionError(
                f"Failed to store {column_name}: {e}", operation_data=op_data
            ) from e

    def store_findings(self, entity_id: str, findings: List[SynthesizedFinding]):
        """Stores synthesized findings for an entity."""
        if findings:  # Avoid storing empty lists unnecessarily
            self._store_attribute(
                entity_id, FINDINGS_COL, [f.model_dump() for f in findings]
            )

    def store_obstacles(self, entity_id: str, obstacles: List[IdentifiedObstacle]):
        """Stores identified obstacles for an entity."""
        if obstacles:
            self._store_attribute(
                entity_id, OBSTACLES_COL, [o.model_dump() for o in obstacles]
            )

    def store_proposals(self, entity_id: str, proposals: List[ProposedNextStep]):
        """Stores proposed next steps for an entity."""
        if proposals:
            self._store_attribute(
                entity_id, PROPOSALS_COL, [p.model_dump() for p in proposals]
            )

    def store_confidence(self, entity_id: str, confidence: Optional[float]):
        """Stores the overall confidence score for an entity's report cycle."""
        self._store_attribute(entity_id, CONFIDENCE_COL, confidence)

    def store_narrative(self, entity_id: str, narrative: Optional[str]):
        """Stores the summary narrative for an entity's report cycle."""
        self._store_attribute(entity_id, NARRATIVE_COL, narrative)

    # --- End Methods for Storing Richer Data ---

    # --- Methods for Retrieving Richer Data / Context ---

    def get_obstacle_summary(self) -> List[Dict[str, Any]]:
        """Retrieves a summary of entities with active obstacles."""
        self._raise_if_not_ready("get_obstacle_summary")
        try:
            df = self._data_store.get_all_data()
            if df.empty or OBSTACLES_COL not in df.columns:
                return []

            obstacle_df = df[df[OBSTACLES_COL].notna()]

            summary = []
            for entity_id, row in obstacle_df.iterrows():
                obstacles_raw = row[OBSTACLES_COL]
                obstacles_list = None
                if isinstance(obstacles_raw, list) and obstacles_raw:
                    obstacles_list = obstacles_raw
                elif obstacles_raw:
                    logger.warning(
                        f"Obstacle data for entity '{entity_id}' is not a list: {type(obstacles_raw)}"
                    )

                if obstacles_list:
                    summary.append(
                        {
                            "entity_id": str(entity_id),  # Ensure ID is string
                            "obstacles": obstacles_list,
                        }
                    )

            return summary
        except Exception as e:
            logger.error(f"Error generating obstacle summary: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="KnowledgeBase",
                operation="get_obstacle_summary",
                details=str(e),
            )
            raise KBInteractionError(
                f"Error generating obstacle summary: {e}", operation_data=op_data
            ) from e

    def get_findings(self, entity_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves stored findings (list of dicts) for a specific entity."""
        self._raise_if_not_ready("get_findings")
        # Assumes _data_store has get_feedback_for_entity
        return self._data_store.get_feedback_for_entity(entity_id, FINDINGS_COL)

    def get_obstacles(self, entity_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves stored obstacles (list of dicts) for a specific entity."""
        self._raise_if_not_ready("get_obstacles")
        # Assumes _data_store has get_feedback_for_entity
        return self._data_store.get_feedback_for_entity(entity_id, OBSTACLES_COL)

    def get_proposals(self, entity_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves stored proposals (list of dicts) for a specific entity."""
        self._raise_if_not_ready("get_proposals")
        # Assumes _data_store has get_feedback_for_entity
        return self._data_store.get_feedback_for_entity(entity_id, PROPOSALS_COL)

    def get_confidence(self, entity_id: str) -> Optional[float]:
        """Retrieves the stored confidence score for a specific entity."""
        self._raise_if_not_ready("get_confidence")
        # Assumes _data_store has get_feedback_for_entity
        return self._data_store.get_feedback_for_entity(entity_id, CONFIDENCE_COL)

    def get_narrative(self, entity_id: str) -> Optional[str]:
        """Retrieves the stored narrative for a specific entity."""
        self._raise_if_not_ready("get_narrative")
        # Assumes _data_store has get_feedback_for_entity
        return self._data_store.get_feedback_for_entity(entity_id, NARRATIVE_COL)

    def get_entities_with_active_obstacles(self) -> List[str]:
        """Retrieves IDs of entities that have non-empty obstacle lists from the DataStore."""
        self._raise_if_not_ready("get_entities_with_active_obstacles")
        try:
            # Assumes _data_store has get_entities_with_active_obstacles
            return self._data_store.get_entities_with_active_obstacles()
        except Exception as e:
            # Log error but return empty list to avoid breaking planner
            logger.error(
                f"Error retrieving entities with obstacles from DataStore: {e}",
                exc_info=True,
            )
            return []

    def get_average_confidence(self) -> Optional[float]:
        """Calculates the average confidence score via the DataStore."""
        self._raise_if_not_ready("get_average_confidence")
        try:
            # Assumes _data_store has get_average_confidence
            return self._data_store.get_average_confidence()
        except Exception as e:
            # Log error but return None to avoid breaking planner
            logger.error(
                f"Error retrieving average confidence from DataStore: {e}",
                exc_info=True,
            )
            return None

    def get_low_confidence_entities(self, threshold: float) -> List[str]:
        """Retrieves IDs of low confidence entities via the DataStore."""
        self._raise_if_not_ready("get_low_confidence_entities")
        try:
            # Assumes _data_store has get_low_confidence_entities
            return self._data_store.get_low_confidence_entities(threshold)
        except Exception as e:
            # Log error but return empty list to avoid breaking planner
            logger.error(
                f"Error retrieving low confidence entities from DataStore: {e}",
                exc_info=True,
            )
            return []

    # --- End Methods for Retrieving Richer Data / Context ---

    def get_entity_profile(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves entity data from the DataStore. Internal columns contain Python objects."""
        self._raise_if_not_ready("get_entity_profile")
        try:
            return self._data_store.get_entity_data(entity_id)
        except Exception as e:
            logger.error(
                f"Failed to get entity profile for '{entity_id}' from DataStore: {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="get_entity_data", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to get entity profile '{entity_id}': {e}",
                operation_data=op_data,
            ) from e

    def get_full_dataset(self) -> pd.DataFrame:
        """Retrieves the full dataset from the DataStore (internal columns contain Python objects)."""
        logger.debug("Entering TabularKnowledgeBase.get_full_dataset")
        self._raise_if_not_ready("get_full_dataset")
        try:
            logger.debug(
                f"Calling self._data_store.get_all_data(). DataStore instance: {self._data_store}"
            )
            df = self._data_store.get_all_data()
            logger.debug(
                f"DataStore get_all_data returned. DataFrame is empty: {df.empty}. Shape: {df.shape if not df.empty else 'N/A'}"
            )
            return df
        except Exception as e:
            logger.error(
                f"Failed to get full dataset from DataStore: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="get_all_data", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to get full dataset: {e}", operation_data=op_data
            ) from e

    def get_data_preview(self, num_rows: int = 10) -> pd.DataFrame:
        """Retrieves a preview (first N rows) of the full dataset."""
        logger.debug(
            f"Entering TabularKnowledgeBase.get_data_preview (num_rows={num_rows})"
        )
        self._raise_if_not_ready("get_data_preview")
        try:
            # Reuse get_full_dataset to ensure consistency and error handling
            full_df = self.get_full_dataset()
            logger.debug(
                f"get_full_dataset returned shape: {full_df.shape if not full_df.empty else 'N/A'}"
            )

            if full_df.empty:
                logger.debug(
                    "Full dataset is empty, returning empty DataFrame for preview."
                )
                return full_df  # Return the empty DataFrame directly

            preview_df = full_df.head(num_rows)
            logger.debug(f"Preview DataFrame created with shape: {preview_df.shape}")
            return preview_df
        except KBInteractionError:
            # Error already logged by get_full_dataset, just re-raise
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error generating data preview (num_rows={num_rows}): {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="KnowledgeBase", operation="get_data_preview", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to get data preview: {e}", operation_data=op_data
            ) from e

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Gets the data summary from the DataStore."""
        self._raise_if_not_ready("get_knowledge_summary")
        try:
            summary = self._data_store.get_data_summary()
            summary["ready"] = True  # Add readiness flag here
            return summary
        except Exception as e:
            logger.error(
                f"Failed to get knowledge summary from DataStore: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="get_data_summary", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to get knowledge summary: {e}", operation_data=op_data
            ) from e

    def find_incomplete_entities(
        self, required_attributes: Optional[List[str]] = None
    ) -> List[str]:
        """Finds incomplete entities using the DataStore (checks user columns)."""
        self._raise_if_not_ready("find_incomplete_entities")

        check_attributes: List[str] = []
        user_cols = []
        try:
            # Need user columns from data store to filter attributes
            user_cols = self._data_store.user_columns
        except Exception as e:
            logger.error(
                f"Failed to get user columns from DataStore for find_incomplete_entities: {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="DataStore", operation="user_columns property", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to get user columns: {e}", operation_data=op_data
            ) from e

        if required_attributes:
            check_attributes = [
                attr for attr in required_attributes if attr in user_cols
            ]
        elif self._task_definition and self._task_definition.target_columns:
            check_attributes = [
                col for col in self._task_definition.target_columns if col in user_cols
            ]
        elif (
            user_cols
            and self._task_definition
            and self._task_definition.identifier_column
        ):
            check_attributes = [
                col
                for col in user_cols
                if col != self._task_definition.identifier_column
            ]

        if not check_attributes:
            logger.warning(
                "No valid user-defined attributes specified or derivable for find_incomplete_entities."
            )
            return []

        try:
            return self._data_store.find_entities_lacking_attributes(check_attributes)
        except Exception as e:
            logger.error(
                f"Failed to find incomplete entities in DataStore: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="DataStore",
                operation="find_entities_lacking_attributes",
                details=str(e),
            )
            raise KBInteractionError(
                f"Failed to find incomplete entities: {e}", operation_data=op_data
            ) from e

    def reset_knowledge(self):
        """Resets the underlying DataStore."""
        logger.warning("Resetting Knowledge Base...")
        try:
            if self._data_store:
                self._data_store.reset_storage()
            self._task_definition = None  # Reset task definition as well
            logger.info("Knowledge Base reset complete.")
        except Exception as e:
            logger.error(f"Failed to reset DataStore: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="DataStore", operation="reset_storage", details=str(e)
            )
            raise KBInteractionError(
                f"Failed to reset KB: {e}", operation_data=op_data
            ) from e
