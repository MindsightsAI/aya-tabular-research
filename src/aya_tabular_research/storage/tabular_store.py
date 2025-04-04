import logging
from typing import Any, Dict, List, Optional

import pandas as pd

# Import custom exceptions and error models
from ..core.exceptions import KBInteractionError
from ..core.models.error_models import OperationalErrorData
from ..core.models.research_models import TaskDefinition
from .data_store import DataStore

logger = logging.getLogger(__name__)

# Define constants for the internal columns used to store richer report data
FINDINGS_COL = "_synthesized_findings"
OBSTACLES_COL = "_identified_obstacles"
PROPOSALS_COL = "_proposed_next_steps"
CONFIDENCE_COL = "_confidence_score"  # For overall report confidence
NARRATIVE_COL = "_summary_narrative"  # To store narrative per entity/report cycle

# List of all internal columns for richer data
INTERNAL_COLS = [
    FINDINGS_COL,
    OBSTACLES_COL,
    PROPOSALS_COL,
    CONFIDENCE_COL,
    NARRATIVE_COL,
]


class TabularStore(DataStore):
    """
    Concrete implementation of DataStore using a Pandas DataFrame for storage.
    Includes internal columns to store serialized richer report data (Phase 3).
    Raises KBInteractionError on operational failures.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._task_definition: Optional[TaskDefinition] = None
        self._identifier_column: Optional[str] = None
        self._columns: List[str] = []  # Will include user columns + internal cols
        self._user_columns: List[str] = []  # Just the columns defined by the user
        self._is_initialized: bool = False
        logger.info("TabularStore initialized.")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def identifier_column(self) -> Optional[str]:
        return self._identifier_column

    @property
    def columns(self) -> List[str]:
        """Returns all columns including internal ones."""
        return self._columns

    @property
    def user_columns(self) -> List[str]:
        """Returns only the columns defined by the user in the task definition."""
        return self._user_columns

    def _raise_if_not_initialized(self, operation: str):
        """Helper to raise KBInteractionError if not initialized."""
        if (
            not self.is_initialized
            or self._df is None
            or self._identifier_column is None
        ):
            msg = f"TabularStore not initialized. Cannot perform operation: {operation}"
            logger.error(msg)
            op_data = OperationalErrorData(
                component="TabularStore",
                operation=operation,
                details="Store not initialized.",
            )
            raise KBInteractionError(msg, operation_data=op_data)

    def initialize_storage(self, task_definition: TaskDefinition):
        """Initializes the DataFrame based on the task definition, adding internal columns."""
        operation = "initialize_storage"
        if self._is_initialized:
            logger.warning(
                "Storage already initialized. Resetting and re-initializing."
            )
            # Allow reset_storage to potentially raise KBInteractionError
            self.reset_storage()

        logger.info(
            f"Initializing TabularStore with user columns: {task_definition.columns}"
        )
        self._task_definition = task_definition
        self._identifier_column = task_definition.identifier_column
        self._user_columns = task_definition.columns

        self._columns = sorted(list(set(self._user_columns + INTERNAL_COLS)))
        dtypes = {col: "object" for col in INTERNAL_COLS}

        try:
            self._df = pd.DataFrame(columns=self._columns).astype(
                dtypes, errors="ignore"
            )
            if self._identifier_column not in self._df.columns:
                # This should not happen if validation passed, but check defensively
                raise ValueError(
                    f"Identifier column '{self._identifier_column}' not found in defined columns."
                )
            self._df = self._df.set_index(self._identifier_column, drop=False)
            self._is_initialized = True
            logger.info(
                f"TabularStore initialized successfully with index '{self._identifier_column}' and columns: {self._columns}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize DataFrame: {e}", exc_info=True)
            self._is_initialized = False
            self._df = None
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed to initialize TabularStore: {e}", operation_data=op_data
            ) from e

    def add_or_update_entity(self, entity_id: str, attributes: Dict[str, Any]):
        """Adds or updates an entity in the DataFrame. Handles user and internal columns."""
        operation = "add_or_update_entity"
        self._raise_if_not_initialized(operation)
        assert self._df is not None  # Ensure type checker knows df is not None
        assert self._identifier_column is not None

        try:
            # Ensure the identifier column value is consistent
            if (
                self._identifier_column in attributes
                and attributes[self._identifier_column] != entity_id
            ):
                logger.warning(
                    f"Mismatch between entity_id ('{entity_id}') and identifier in attributes ('{attributes[self._identifier_column]}'). Using entity_id."
                )
                attributes[self._identifier_column] = entity_id
            elif self._identifier_column not in attributes:
                attributes[self._identifier_column] = entity_id

            valid_attributes = {
                k: v for k, v in attributes.items() if k in self._columns
            }
            unknown_attrs = set(attributes.keys()) - set(valid_attributes.keys())
            if unknown_attrs:
                logger.debug(
                    f"Ignoring unknown attributes for entity '{entity_id}': {unknown_attrs}"
                )

            entity_series = pd.Series(valid_attributes)

            if entity_id in self._df.index:
                logger.debug(
                    f"Updating entity: {entity_id} with attributes: {list(valid_attributes.keys())}"
                )
                # Use .loc for potentially setting new columns if needed, though update is safer
                for col, value in entity_series.items():
                    if (
                        col in self._df.columns
                    ):  # Ensure column exists before assignment
                        self._df.loc[entity_id, col] = value
                    else:
                        logger.warning(
                            f"Attempted to update non-existent column '{col}' for entity '{entity_id}'. Skipping."
                        )
                # Alternative using update (safer, only updates existing columns):
                # update_df = pd.DataFrame([valid_attributes]).set_index(self._identifier_column)
                # self._df.update(update_df)
            else:
                logger.debug(f"Adding new entity: {entity_id}")
                # Create a DataFrame for the new row to ensure proper alignment
                new_row_df = pd.DataFrame([valid_attributes])
                new_row_df = new_row_df.set_index(self._identifier_column, drop=False)
                # Reindex to ensure column order and presence matches the main DataFrame
                new_row_df = new_row_df.reindex(columns=self._df.columns)
                self._df = pd.concat([self._df, new_row_df], ignore_index=False)

        except Exception as e:
            logger.error(f"Error in {operation} for '{entity_id}': {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore",
                operation=operation,
                details=f"Entity ID '{entity_id}': {e}",
            )
            raise KBInteractionError(
                f"Failed operation '{operation}' for entity '{entity_id}': {e}",
                operation_data=op_data,
            ) from e

    def batch_add_or_update(self, entities: List[Dict[str, Any]]):
        """Adds or updates multiple entities efficiently."""
        operation = "batch_add_or_update"
        self._raise_if_not_initialized(operation)
        assert self._df is not None  # Ensure type checker knows df is not None
        assert self._identifier_column is not None

        if not entities:
            logger.debug(f"{operation} called with empty list.")
            return

        num_incoming = len(entities)
        logger.debug(f"Starting {operation} for {num_incoming} entities.")

        try:
            processed_entities = []
            malformed_count = 0
            for entity_dict in entities:
                entity_id = entity_dict.get(self._identifier_column)
                if entity_id is None or pd.isnull(entity_id):
                    logger.warning(
                        f"Skipping entity in batch due to missing/null identifier: {entity_dict}"
                    )
                    malformed_count += 1
                    continue
                entity_id_str = str(entity_id)

                valid_data = {
                    k: v for k, v in entity_dict.items() if k in self._columns
                }
                valid_data[self._identifier_column] = entity_id_str
                processed_entities.append(valid_data)

            if not processed_entities:
                logger.warning(
                    f"No valid entities remaining after preprocessing {num_incoming} inputs."
                )
                return

            updates_df = pd.DataFrame(processed_entities)
            updates_df = updates_df.set_index(self._identifier_column, drop=False)

            existing_ids = self._df.index.intersection(updates_df.index)
            new_ids = updates_df.index.difference(self._df.index)

            new_entities_df = updates_df.loc[new_ids]
            existing_entities_df = updates_df.loc[existing_ids]

            added_count = 0
            if not new_entities_df.empty:
                logger.debug(f"Adding {len(new_entities_df)} new entities in batch.")
                new_entities_df = new_entities_df.reindex(columns=self._df.columns)
                self._df = pd.concat([self._df, new_entities_df], ignore_index=False)
                added_count = len(new_entities_df)

            updated_count = 0
            if not existing_entities_df.empty:
                logger.debug(
                    f"Updating {len(existing_entities_df)} existing entities in batch."
                )
                # Ensure columns match before update to avoid adding new columns unexpectedly
                update_cols = existing_entities_df.columns.intersection(
                    self._df.columns
                )
                self._df.update(existing_entities_df[update_cols])
                updated_count = len(existing_entities_df)

            logger.info(
                f"{operation} complete. Added: {added_count}, Updated: {updated_count}, Malformed/Skipped: {malformed_count}"
            )

        except Exception as e:
            logger.error(f"Error during {operation}: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e

    def add_entities(self, entities: List[Dict[str, Any]]):
        """Adds multiple NEW entities, skipping existing ones. Uses batch_add_or_update."""
        operation = "add_entities"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert self._identifier_column is not None

        if not entities:
            logger.debug(f"{operation} called with empty list.")
            return

        logger.debug(
            f"Processing {operation} request for {len(entities)} potential entities."
        )
        try:
            existing_ids = self._df.index
            new_entities_to_add = []
            skipped_count = 0
            for entity in entities:
                entity_id = entity.get(self._identifier_column)
                if (
                    entity_id is not None
                    and not pd.isnull(entity_id)
                    and str(entity_id) in existing_ids
                ):
                    skipped_count += 1
                else:
                    new_entities_to_add.append(entity)

            if skipped_count > 0:
                logger.warning(
                    f"{operation}: Skipped {skipped_count} entities because their IDs already exist."
                )

            if not new_entities_to_add:
                logger.info(
                    f"{operation}: No new entities to add after filtering duplicates."
                )
                return

            # Delegate to batch method
            self.batch_add_or_update(new_entities_to_add)
        except KBInteractionError:  # Catch error from batch_add_or_update
            logger.error(f"Error during {operation} via batch method.", exc_info=False)
            raise  # Re-raise
        except Exception as e:  # Catch other unexpected errors
            logger.error(f"Unexpected error during {operation}: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e

    def get_entity_data(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves data for a specific entity."""
        operation = "get_entity_data"
        self._raise_if_not_initialized(operation)
        assert self._df is not None

        try:
            if entity_id in self._df.index:
                # Use .loc[entity_id].to_dict() after handling NaNs
                entity_series = self._df.loc[entity_id]
                # Replace Pandas NaT/NaN with None for consistent dict representation
                entity_data = entity_series.where(
                    pd.notna(entity_series), None
                ).to_dict()
                return entity_data
            else:
                logger.debug(f"Entity '{entity_id}' not found.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving entity '{entity_id}': {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore",
                operation=operation,
                details=f"Entity ID '{entity_id}': {e}",
            )
            raise KBInteractionError(
                f"Failed operation '{operation}' for entity '{entity_id}': {e}",
                operation_data=op_data,
            ) from e

    def get_all_data(self) -> pd.DataFrame:
        """Retrieves all data as a DataFrame."""
        operation = "get_all_data"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        try:
            return self._df.copy()
        except Exception as e:
            logger.error(f"Error retrieving all data: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e

    def get_data_summary(self) -> Dict[str, Any]:
        """Provides a summary of the data."""
        operation = "get_data_summary"
        if not self.is_initialized or self._df is None:
            # Don't raise error here, just return uninitialized summary
            return {
                "initialized": False,
                "entity_count": 0,
                "columns": [],
                "user_columns": [],
            }

        try:
            entity_count = len(self._df)
            # Handle potential error during sum() if columns have incompatible types after storing objects
            try:
                non_null_counts = self._df.notna().sum().to_dict()
            except TypeError as te:
                logger.warning(
                    f"TypeError during non-null count summary, possibly due to mixed types in object columns: {te}"
                )
                non_null_counts = {
                    col: self._df[col].notna().sum() for col in self._df.columns
                }

            return {
                "initialized": True,
                "entity_count": entity_count,
                "columns": self._columns,
                "user_columns": self._user_columns,
                "identifier_column": self._identifier_column,
                "non_null_counts": non_null_counts,
            }
        except Exception as e:
            logger.error(f"Error generating data summary: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e

    def find_entities_lacking_attributes(
        self, required_attributes: List[str]
    ) -> List[str]:
        """Finds entities missing required attributes (checks only user columns)."""
        operation = "find_entities_lacking_attributes"
        self._raise_if_not_initialized(operation)
        assert self._df is not None

        if self._df.empty:
            return []

        try:
            valid_required = [
                attr for attr in required_attributes if attr in self._user_columns
            ]
            if not valid_required:
                logger.warning(
                    "No valid user-defined required attributes provided for completion check."
                )
                return []

            # Check if all required columns actually exist in the DataFrame
            missing_cols = set(valid_required) - set(self._df.columns)
            if missing_cols:
                logger.error(
                    f"Required attributes {missing_cols} not found in DataFrame columns for checking."
                )
                # Decide whether to raise or return empty list. Raising seems safer.
                op_data = OperationalErrorData(
                    component="TabularStore",
                    operation=operation,
                    details=f"Required check columns missing: {missing_cols}",
                )
                raise KBInteractionError(
                    f"Cannot check for missing attributes, columns not found: {missing_cols}",
                    operation_data=op_data,
                )

            mask = self._df[valid_required].isnull().any(axis=1)
            missing_entities = self._df[mask].index.tolist()

            logger.debug(
                f"Found {len(missing_entities)} entities lacking one or more of user attributes: {valid_required}."
            )
            return [str(eid) for eid in missing_entities]  # Ensure IDs are strings
        except Exception as e:
            logger.error(
                f"Error finding entities lacking attributes: {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e

    def reset_storage(self):
        """Resets the store. Re-initializes if task definition exists."""
        operation = "reset_storage"
        logger.warning("Resetting TabularStore.")

        try:
            self._df = None
            self._task_definition = None
            self._identifier_column = None
            self._columns = []
            self._user_columns = []
            self._is_initialized = False
            # Re-initialization attempt is removed from here.
            # The caller (e.g., KnowledgeBase) should handle re-initialization if needed.
            logger.info("TabularStore reset to uninitialized state.")
        except Exception as e:
            logger.error(f"Error during reset_storage: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e
