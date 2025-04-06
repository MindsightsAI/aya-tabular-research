import logging
from typing import Any, Dict, List, Optional, Union

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
    Handles dynamic indexing based on TaskDefinition.granularity_columns.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._task_definition: Optional[TaskDefinition] = None
        self._identifier_column: Optional[str] = None
        self._columns: List[str] = []  # Will include user columns + internal cols
        self._user_columns: List[str] = []  # Just the columns defined by the user
        self._index_columns: List[str] = []  # Columns used for the DataFrame index
        self._is_initialized: bool = False
        logger.info("TabularStore initialized.")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def identifier_column(self) -> Optional[str]:
        """The primary entity identifier column name."""
        return self._identifier_column

    @property
    def columns(self) -> List[str]:
        """Returns all columns including internal ones."""
        return self._columns

    @property
    def user_columns(self) -> List[str]:
        """Returns only the columns defined by the user in the task definition."""
        return self._user_columns

    @property
    def index_columns(self) -> List[str]:
        """Returns the columns currently used as the index."""
        return self._index_columns

    def _raise_if_not_initialized(self, operation: str):
        """Helper to raise KBInteractionError if not initialized."""
        if (
            not self.is_initialized
            or self._df is None
            or not self._index_columns  # Check if index columns are set
            # identifier_column is needed for feedback logic, check if it's set when needed
        ):
            # Specific check for feedback operations requiring identifier_column
            is_feedback_op = (
                any(col in operation for col in INTERNAL_COLS)
                or "feedback" in operation
                or "clear_obstacles" in operation
            )
            if is_feedback_op and self._identifier_column is None:
                msg = f"TabularStore not initialized or identifier_column not set for feedback operation: {operation}"
            elif not self.is_initialized or self._df is None or not self._index_columns:
                msg = f"TabularStore not initialized. Cannot perform operation: {operation}"
            else:  # Should not be reached if logic is correct, but as fallback
                msg = f"TabularStore in unexpected state for operation: {operation}"

            logger.error(msg)
            op_data = OperationalErrorData(
                component="TabularStore",
                operation=operation,
                details="Store not initialized or missing required configuration (index/identifier).",
            )
            raise KBInteractionError(msg, operation_data=op_data)

    def initialize_storage(self, task_definition: TaskDefinition):
        """Initializes the DataFrame based on the task definition, adding internal columns."""
        operation = "initialize_storage"
        if self._is_initialized:
            logger.warning(
                "Storage already initialized. Resetting and re-initializing."
            )
            self.reset_storage()  # Allow reset_storage to potentially raise

        logger.info(
            f"Initializing TabularStore with user columns: {task_definition.columns}"
        )
        self._task_definition = task_definition
        self._identifier_column = task_definition.identifier_column
        self._user_columns = task_definition.columns

        # Determine index columns based on granularity_columns
        if (
            task_definition.granularity_columns
            and self._identifier_column  # Ensure identifier is defined if granularity is used
            and self._identifier_column in task_definition.granularity_columns
        ):
            # Ensure uniqueness and order consistency
            self._index_columns = sorted(list(set(task_definition.granularity_columns)))
            logger.info(f"Using granularity columns for index: {self._index_columns}")
        elif (
            self._identifier_column
        ):  # Default to identifier if defined and no valid granularity
            self._index_columns = [self._identifier_column]
            logger.info(f"Using identifier column for index: {self._index_columns}")
        else:
            # This case should ideally be prevented by TaskDefinition validation
            raise ValueError(
                "Cannot initialize TabularStore: identifier_column is required if granularity_columns are not provided or invalid."
            )

        self._columns = sorted(list(set(self._user_columns + INTERNAL_COLS)))
        dtypes = {col: "object" for col in INTERNAL_COLS}

        try:
            self._df = pd.DataFrame(columns=self._columns).astype(
                dtypes, errors="ignore"
            )
            # Ensure all index columns exist in the defined columns
            missing_index_cols = [
                col for col in self._index_columns if col not in self._df.columns
            ]
            if missing_index_cols:
                raise ValueError(
                    f"Index columns {missing_index_cols} not found in defined columns {self._columns}."
                )

            # Set the index using the determined columns
            self._df = self._df.set_index(self._index_columns, drop=False)
            self._is_initialized = True
            logger.info(
                f"TabularStore initialized successfully with index {self._index_columns} and columns: {self._columns}"
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
        """
        DEPRECATED - Use batch_add_or_update.
        Adds or updates a single row defined by its index values.
        NOTE: This method is less efficient and primarily intended for targeted
              feedback updates (like clearing obstacles) applied via batch logic.
        """
        operation = "add_or_update_entity (DEPRECATED - use batch)"
        logger.warning(f"{operation} called directly. Prefer batch_add_or_update.")
        self._raise_if_not_initialized(operation)
        assert self._df is not None

        try:
            # Construct the index tuple/value from the attributes
            try:
                # Ensure all index columns are present in attributes
                index_val_list = [attributes[col] for col in self._index_columns]
                index_val = (
                    tuple(index_val_list)
                    if len(self._index_columns) > 1
                    else index_val_list[0]
                )
            except KeyError as e:
                raise ValueError(
                    f"Missing index column '{e}' in attributes for update."
                ) from e

            # Filter attributes to only include valid columns
            valid_attributes = {
                k: v for k, v in attributes.items() if k in self._columns
            }

            # Use batch update for consistency
            self.batch_add_or_update([valid_attributes])

        except Exception as e:
            index_val_repr = "unknown"
            if "index_val" in locals():
                index_val_repr = repr(index_val)
            logger.error(
                f"Error in {operation} for index '{index_val_repr}': {e}",
                exc_info=True,
            )
            op_data = OperationalErrorData(
                component="TabularStore",
                operation=operation,
                details=f"Index '{index_val_repr}': {e}",
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}",
                operation_data=op_data,
            ) from e

    def batch_add_or_update(self, entities: List[Dict[str, Any]]):
        """Adds or updates multiple entities (rows) efficiently based on the defined index (`self._index_columns`)."""
        operation = "batch_add_or_update"
        self._raise_if_not_initialized(operation)
        assert self._df is not None

        if not entities:
            logger.debug(f"{operation} called with empty list.")
            return

        num_incoming = len(entities)
        logger.debug(f"Starting {operation} for {num_incoming} entities.")

        try:
            processed_entities_map = (
                {}
            )  # Use dict to handle duplicates within batch easily
            malformed_count = 0

            for entity_dict in entities:
                # Construct index key, allowing pd.NA for non-identifier index columns
                index_val_list = []
                try:
                    for col in self._index_columns:
                        val = entity_dict.get(col)  # Use .get() to avoid KeyError
                        if col == self._identifier_column:
                            if pd.isnull(val):
                                raise ValueError(
                                    f"Identifier column '{col}' cannot be null"
                                )
                            index_val_list.append(val)
                        elif pd.isnull(val):
                            index_val_list.append(
                                pd.NA
                            )  # Use pd.NA for missing non-identifier index cols
                        else:
                            index_val_list.append(val)
                    index_key = tuple(index_val_list)
                except ValueError as e:  # Catches null identifier error
                    logger.warning(
                        f"Skipping entity in batch due to null identifier column value ({e}): {entity_dict}"
                    )
                    malformed_count += 1
                    continue
                # Note: KeyError is no longer possible here due to .get()

                # Filter data and ensure index columns (potentially with pd.NA) are included
                valid_data = {
                    k: v for k, v in entity_dict.items() if k in self._columns
                }
                for i, col in enumerate(self._index_columns):
                    valid_data[col] = index_val_list[i]  # Assign potentially NA value

                # Overwrite previous entry in batch if duplicate index found
                if index_key in processed_entities_map:
                    logger.debug(
                        f"Duplicate index {index_key} found within batch. Keeping last occurrence."
                    )
                processed_entities_map[index_key] = valid_data

            processed_entities = list(processed_entities_map.values())

            if not processed_entities:
                logger.warning(
                    f"No valid entities remaining after preprocessing {num_incoming} inputs."
                )
                return

            # Create DataFrame from processed data and set the correct index
            updates_df = pd.DataFrame(processed_entities)
            # Important: Ensure index columns have compatible types with the main DataFrame index
            for _, col in enumerate(self._index_columns):
                if col in updates_df.columns and col in self._df.index.names:
                    target_dtype = self._df.index.get_level_values(col).dtype
                    if updates_df[col].dtype != target_dtype:
                        try:
                            updates_df[col] = updates_df[col].astype(target_dtype)
                        except Exception as e:
                            logger.warning(
                                f"Could not convert index column '{col}' to type {target_dtype}: {e}"
                            )
            updates_df = updates_df.set_index(self._index_columns, drop=False)

            # Identify new vs existing rows based on the index
            existing_ids = self._df.index.intersection(updates_df.index)
            new_ids = updates_df.index.difference(self._df.index)

            # Use index directly for selection
            new_entities_df = (
                updates_df.loc[new_ids]
                if not new_ids.empty
                else pd.DataFrame(columns=updates_df.columns).set_index(
                    self._index_columns, drop=False
                )
            )
            existing_entities_df = (
                updates_df.loc[existing_ids]
                if not existing_ids.empty
                else pd.DataFrame(columns=updates_df.columns).set_index(
                    self._index_columns, drop=False
                )
            )

            # Add new rows
            added_count = 0
            if not new_entities_df.empty:
                logger.debug(f"Adding {len(new_entities_df)} new entities in batch.")
                # Ensure columns match before concat
                new_entities_df = new_entities_df.reindex(columns=self._df.columns)
                self._df = pd.concat(  # type: ignore # Ignore because concat can handle None df
                    [self._df, new_entities_df], ignore_index=False, sort=False
                )
                added_count = len(new_entities_df)

            # Update existing rows
            updated_count = 0
            if not existing_entities_df.empty:
                logger.debug(
                    f"Updating {len(existing_entities_df)} existing entities in batch."
                )
                # Separate feedback columns from regular data columns
                update_cols = existing_entities_df.columns.intersection(
                    self._df.columns
                )
                feedback_cols_to_update = [
                    col for col in update_cols if col in INTERNAL_COLS
                ]
                non_feedback_cols_to_update = [
                    col for col in update_cols if col not in INTERNAL_COLS
                ]

                # Update non-feedback columns directly using the full index
                if non_feedback_cols_to_update:
                    # Use update for efficiency, it aligns on index automatically
                    # Ensure dtypes match where possible before update to avoid warnings/errors
                    for col in non_feedback_cols_to_update:
                        if (
                            col in self._df.columns
                            and self._df[col].dtype != existing_entities_df[col].dtype
                        ):
                            try:
                                # Attempt conversion safely
                                converted_col = existing_entities_df[col].astype(
                                    self._df[col].dtype, errors="ignore"
                                )
                                # Only update if conversion didn't change dtype unexpectedly (e.g., object -> object is fine)
                                if converted_col.dtype == self._df[col].dtype:
                                    existing_entities_df[col] = converted_col
                                else:
                                    # If dtype conversion failed or resulted in incompatible type, skip direct update for this col
                                    logger.debug(
                                        f"Skipping dtype alignment for column '{col}' due to incompatible types."
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not align dtype for column '{col}': {e}. Update might raise warning/error."
                                )
                                pass  # Ignore conversion errors, update will handle or raise warning
                    self._df.update(existing_entities_df[non_feedback_cols_to_update])

                # Update feedback columns based on the primary identifier (_identifier_column)
                # Apply the *last* feedback value from the batch for a given entity_id to all rows of that entity_id
                if feedback_cols_to_update and self._identifier_column:
                    identifier_level_name = self._identifier_column
                    is_multi_index = isinstance(self._df.index, pd.MultiIndex)

                    # Check if identifier column is part of the index
                    if (
                        is_multi_index
                        and self._identifier_column not in self._df.index.names
                    ) or (
                        not is_multi_index
                        and self._identifier_column != self._df.index.name
                    ):
                        logger.error(
                            f"Identifier column '{self._identifier_column}' not found in index. Cannot apply feedback updates."
                        )
                    else:
                        # --- Refactored Feedback Update Logic ---
                        logger.debug(
                            f"Applying feedback updates for {len(existing_ids)} existing rows using vectorized approach."
                        )

                        # 1. Get the last feedback values from the batch per entity_id
                        if is_multi_index:
                            last_feedback_map = (
                                existing_entities_df[feedback_cols_to_update]
                                .groupby(level=identifier_level_name, sort=False)
                                .last()
                            )
                        else:  # Single index
                            # Need to handle potential duplicate index entries in the batch explicitly
                            last_feedback_map = existing_entities_df[
                                ~existing_entities_df.index.duplicated(keep="last")
                            ][feedback_cols_to_update]

                        # 2. Map these values to the target rows in the main DataFrame
                        target_rows = self._df.loc[
                            existing_ids
                        ]  # Select the rows to update in self._df

                        if is_multi_index:
                            target_identifiers = target_rows.index.get_level_values(
                                identifier_level_name
                            )
                        else:  # Single index
                            target_identifiers = target_rows.index

                        # 3. Apply updates column by column
                        for fb_col in feedback_cols_to_update:
                            if fb_col not in self._df.columns:
                                logger.warning(
                                    f"Feedback column '{fb_col}' not found in main DataFrame. Skipping update."
                                )
                                continue
                            if fb_col not in last_feedback_map.columns:
                                logger.warning(
                                    f"Feedback column '{fb_col}' not found in batch's last values. Skipping update."
                                )
                                continue

                            # Create the mapping series
                            update_values = target_identifiers.map(
                                last_feedback_map[fb_col]
                            )

                            # Apply the update only where the mapped value is not NaN
                            # (handles cases where an entity_id in self._df might not have had a feedback update in the batch)
                            valid_updates_mask = update_values.notna()
                            if valid_updates_mask.any():
                                # Use .loc with the original index slice (existing_ids) and the mask derived from mapped values
                                # Ensure the update_values series is aligned with the target slice index
                                self._df.loc[
                                    target_rows.index[valid_updates_mask], fb_col
                                ] = update_values[valid_updates_mask]
                                logger.debug(
                                    f"Applied updates for feedback column '{fb_col}' to {valid_updates_mask.sum()} rows."
                                )
                            else:
                                logger.debug(
                                    f"No valid updates found for feedback column '{fb_col}'."
                                )
                        # --- End Refactored Logic ---
                updated_count = len(
                    existing_entities_df
                )  # Count based on unique index rows updated

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
        """Adds multiple NEW entities (rows), skipping existing ones based on index. Uses batch_add_or_update."""
        operation = "add_entities"
        self._raise_if_not_initialized(operation)
        assert self._df is not None

        if not entities:
            logger.debug(f"{operation} called with empty list.")
            return

        logger.debug(
            f"Processing {operation} request for {len(entities)} potential entities."
        )
        try:
            existing_ids = self._df.index  # This is now potentially a MultiIndex
            new_entities_to_add = []
            skipped_count = 0
            malformed_count = 0
            entity_indices_in_batch = (
                set()
            )  # Track indices added in this batch to avoid duplicates from input list

            for entity in entities:
                # Construct the potential index tuple/value from the entity dict
                try:
                    index_values = tuple(
                        entity[col] for col in self._index_columns
                    )  # Always create tuple
                    if any(pd.isnull(val) for val in index_values):
                        raise ValueError("Null value in index column")
                    # Convert tuple to single value if single index for comparison
                    entity_index = (
                        index_values
                        if isinstance(self._df.index, pd.MultiIndex)
                        else index_values[0]
                    )

                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Skipping entity in {operation} due to missing/null index columns ({e}): {entity}"
                    )
                    malformed_count += 1
                    continue

                # Skip if index already exists in DataFrame or previously added in this batch
                if (
                    entity_index in existing_ids
                    or entity_index in entity_indices_in_batch
                ):
                    if (
                        entity_index not in entity_indices_in_batch
                    ):  # Only log skip once per index
                        skipped_count += 1
                else:
                    new_entities_to_add.append(entity)
                    entity_indices_in_batch.add(entity_index)

            if skipped_count > 0:
                logger.info(  # Use info level as this is expected behaviour
                    f"{operation}: Skipped {skipped_count} entities because their index combination already exists."
                )
            if malformed_count > 0:
                logger.warning(
                    f"{operation}: Skipped {malformed_count} entities due to missing/invalid index values."
                )

            if not new_entities_to_add:
                logger.info(f"{operation}: No new entities to add after filtering.")
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

    def get_entity_data(
        self, entity_id: str  # This refers to the value in the identifier_column
    ) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
        """
        Retrieves data associated with a specific primary entity identifier.
        Returns a dict if the store uses a single index (which must be the identifier_column).
        Returns a DataFrame slice containing all rows for that entity if the store uses a MultiIndex.
        Returns None if the entity_id is not found.
        """
        operation = "get_entity_data"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert (
            self._identifier_column is not None
        )  # Need identifier to know which level/index to check

        try:
            if isinstance(self._df.index, pd.MultiIndex):
                # Ensure the identifier column is actually part of the MultiIndex
                if self._identifier_column not in self._df.index.names:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' not found in MultiIndex names: {self._df.index.names}. Cannot retrieve entity data by primary ID."
                    )
                    return None

                identifier_level_name = self._identifier_column
                # Check if the entity_id exists in the correct level of the MultiIndex
                if entity_id in self._df.index.get_level_values(identifier_level_name):
                    # Select all rows matching the entity_id in the correct level
                    # Use xs for potentially better performance and clarity on level selection
                    # Use drop_level=False to keep the index structure consistent
                    entity_data_slice = self._df.xs(
                        entity_id, level=identifier_level_name, drop_level=False
                    )
                    # Return a copy to prevent modification of the original DataFrame slice
                    return entity_data_slice.copy()
                else:
                    logger.debug(
                        f"Entity '{entity_id}' not found in MultiIndex level '{identifier_level_name}'."
                    )
                    return None
            else:  # Single index case
                if (
                    self._identifier_column != self._df.index.name
                ):  # Verify single index is the identifier
                    logger.error(
                        f"Identifier column '{self._identifier_column}' does not match single index name '{self._df.index.name}'. Cannot retrieve entity data."
                    )
                    return None

                if entity_id in self._df.index:
                    # Use .loc which returns a Series for a single match
                    entity_series = self._df.loc[entity_id]
                    # Handle case where single index might still have duplicates (though less common)
                    if isinstance(entity_series, pd.DataFrame):
                        logger.warning(
                            f"Multiple rows found for single index '{entity_id}'. Returning data for the first row."
                        )
                        entity_series = entity_series.iloc[0]

                    # Replace Pandas NaT/NaN with None for consistent dict representation
                    entity_data = entity_series.where(
                        pd.notna(entity_series), None
                    ).to_dict()
                    return entity_data
                else:
                    logger.debug(f"Entity '{entity_id}' not found in single index.")
                    return None
        except (
            KeyError
        ):  # Handle cases where entity_id might not be found cleanly by 'in' check or xs
            logger.debug(f"Entity '{entity_id}' not found (KeyError during access).")
            return None
        except Exception as e:
            logger.error(
                f"Error retrieving entity data for '{entity_id}': {e}", exc_info=True
            )
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
        # operation = "get_data_summary" # Removed unused variable
        if not self.is_initialized or self._df is None:
            # Don't raise error here, just return uninitialized summary
            return {
                "initialized": False,
                "entity_count": 0,
                "row_count": 0,
                "columns": [],
                "user_columns": [],
                "index_columns": [],
                "identifier_column": None,
                "non_null_counts": {},
            }

        try:
            # For MultiIndex, entity_count based on unique primary identifier might be more useful
            row_count = len(self._df)
            entity_count = 0
            if self._identifier_column:
                if (
                    isinstance(self._df.index, pd.MultiIndex)
                    and self._identifier_column in self._df.index.names
                ):
                    entity_count = self._df.index.get_level_values(
                        self._identifier_column
                    ).nunique()
                elif (
                    self._identifier_column == self._df.index.name
                ):  # Single index matches identifier
                    entity_count = self._df.index.nunique()
                else:  # Identifier not in index? Fallback to unique index count
                    entity_count = (
                        len(self._df.index.unique()) if not self._df.empty else 0
                    )
            else:  # No identifier column defined? Fallback to unique index count
                entity_count = len(self._df.index.unique()) if not self._df.empty else 0

            # Handle potential error during sum() if columns have incompatible types after storing objects
            try:
                # Calculate non-null counts per column across all rows
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
                "entity_count": entity_count,  # Count of unique primary entities (or unique index combinations if no identifier)
                "row_count": row_count,  # Total number of rows (granularity points)
                "columns": self._columns,
                "user_columns": self._user_columns,
                "identifier_column": self._identifier_column,  # The designated primary entity identifier
                "index_columns": self._index_columns,  # Report the actual index columns used
                "non_null_counts": non_null_counts,  # Counts per column across all rows
            }
        except Exception as e:
            logger.error(f"Error generating data summary: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore", operation="get_data_summary", details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation 'get_data_summary': {e}", operation_data=op_data
            ) from e

    # --- Methods for Retrieving Richer Feedback Data ---
    # These methods operate at the primary entity level (identifier_column).
    # They retrieve the value from the *first* row associated with the entity_id for simplicity,
    # assuming feedback is applied consistently across all rows for that entity.

    def get_feedback_for_entity(self, entity_id: str, column_name: str) -> Any:
        """
        Helper to retrieve data from a specific feedback column for an entity.
        Retrieves from the first row matching the primary entity_id.
        """
        operation = f"get_feedback_for_entity ({column_name})"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert (
            self._identifier_column is not None
        )  # Feedback is tied to the primary identifier

        if column_name not in INTERNAL_COLS:
            logger.warning(
                f"Attempted to get feedback from non-internal column: {column_name}"
            )
            return None

        try:
            first_matching_row = None
            # Select rows matching the primary entity ID
            if isinstance(self._df.index, pd.MultiIndex):
                # Ensure identifier column is part of the index
                if self._identifier_column not in self._df.index.names:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' not found in MultiIndex names: {self._df.index.names}. Cannot retrieve feedback."
                    )
                    return None
                identifier_level_name = self._identifier_column

                if entity_id in self._df.index.get_level_values(identifier_level_name):
                    # Get the first row matching the entity_id using iloc[0] after selection
                    matching_rows = self._df[
                        self._df.index.get_level_values(identifier_level_name)
                        == entity_id
                    ]
                    if not matching_rows.empty:
                        first_matching_row = matching_rows.iloc[0]
                # else: entity_id not found, first_matching_row remains None

            else:  # Single index
                if self._identifier_column != self._df.index.name:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' does not match single index name '{self._df.index.name}'. Cannot retrieve feedback."
                    )
                    return None
                if entity_id in self._df.index:
                    # Handle potential duplicate single indices
                    result = self._df.loc[entity_id]
                    first_matching_row = (
                        result.iloc[0] if isinstance(result, pd.DataFrame) else result
                    )
                # else: entity_id not found, first_matching_row remains None

            # If no matching row was found
            if first_matching_row is None:
                logger.debug(f"Entity '{entity_id}' not found for feedback retrieval.")
                return None

            # Check if the column exists in the selected row/series
            if column_name in first_matching_row.index:
                value = first_matching_row[column_name]
                # Return None if value is NaN/NaT, otherwise return the value
                # Handle potential Series if entity has multiple rows
                if isinstance(value, pd.Series):
                    # Find the first non-null value in the Series for this entity
                    non_null_values = value.dropna()
                    # Return the first non-null item, or None if all are null
                    return (
                        non_null_values.iloc[0] if not non_null_values.empty else None
                    )
                else:
                    # Original logic for single value
                    return None if pd.isna(value) else value
            else:
                logger.debug(
                    f"Column '{column_name}' not found for entity '{entity_id}' feedback."
                )
                return None

        except KeyError:
            logger.debug(
                f"Entity '{entity_id}' not found (KeyError during feedback access)."
            )
            return None
        except (
            IndexError
        ):  # Handle case where selection results in empty DataFrame (should be caught by empty check now)
            logger.debug(
                f"Entity '{entity_id}' found, but no rows matched for feedback retrieval (IndexError)."
            )
            return None
        except Exception as e:
            logger.error(
                f"Error retrieving feedback '{column_name}' for entity '{entity_id}': {e}",
                exc_info=True,
            )
            # Avoid raising KBInteractionError here? Log and return None.
            return None

    def get_obstacles_for_entity(
        self, entity_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieves identified obstacles for a specific entity."""
        return self.get_feedback_for_entity(entity_id, OBSTACLES_COL)

    def get_confidence_for_entity(self, entity_id: str) -> Optional[float]:
        """Retrieves confidence score for a specific entity."""
        return self.get_feedback_for_entity(entity_id, CONFIDENCE_COL)

    def get_proposals_for_entity(
        self, entity_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieves proposed next steps for a specific entity."""
        return self.get_feedback_for_entity(entity_id, PROPOSALS_COL)

    def get_summary_narrative_for_entity(self, entity_id: str) -> Optional[str]:
        """Retrieves summary narrative for a specific entity."""
        return self.get_feedback_for_entity(entity_id, NARRATIVE_COL)

    def get_entities_with_active_obstacles(self) -> List[str]:
        """Retrieves IDs of primary entities that have non-empty obstacle lists in any of their rows."""
        operation = "get_entities_with_active_obstacles"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert (
            self._identifier_column is not None
        )  # Need identifier to return correct IDs

        if OBSTACLES_COL not in self._df.columns or self._df.empty:
            return []

        try:
            # Filter rows where the obstacle column is not null/NaN AND is not an empty list
            mask = self._df[OBSTACLES_COL].notna() & (
                self._df[OBSTACLES_COL].apply(
                    lambda x: isinstance(x, list) and len(x) > 0
                )
            )
            if not mask.any():
                return []

            # Get the index of the matching rows
            matching_index = self._df[mask].index

            # Extract unique primary entity IDs from the index
            if isinstance(matching_index, pd.MultiIndex):
                if self._identifier_column not in matching_index.names:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' not found in MultiIndex names: {matching_index.names}. Cannot get entities with obstacles."
                    )
                    return []
                unique_entity_ids = (
                    matching_index.get_level_values(self._identifier_column)
                    .unique()
                    .tolist()
                )
            else:  # Single index
                if self._identifier_column != matching_index.name:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' does not match single index name '{matching_index.name}'. Cannot get entities with obstacles."
                    )
                    return []
                unique_entity_ids = matching_index.unique().tolist()

            return [str(eid) for eid in unique_entity_ids]  # Ensure IDs are strings
        except Exception as e:
            logger.error(f"Error in {operation}: {e}", exc_info=True)
            # Avoid raising? Log and return empty list.
            return []

    def get_average_confidence(self) -> Optional[float]:
        """Calculates the average confidence score across rows with scores."""
        operation = "get_average_confidence"
        self._raise_if_not_initialized(operation)
        assert self._df is not None

        if (
            CONFIDENCE_COL not in self._df.columns
            or self._df[CONFIDENCE_COL].isna().all()
        ):
            return None
        try:
            # Calculate mean ignoring NaNs across all relevant rows
            return self._df[CONFIDENCE_COL].mean(skipna=True)
        except Exception as e:
            logger.error(f"Error calculating average confidence: {e}", exc_info=True)
            # Avoid raising? Log and return None.
            return None

    def get_low_confidence_entities(self, threshold: float) -> List[str]:
        """Retrieves IDs of primary entities with any row having confidence score below the threshold."""
        operation = "get_low_confidence_entities"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert (
            self._identifier_column is not None
        )  # Need identifier to return correct IDs

        if CONFIDENCE_COL not in self._df.columns or self._df.empty:
            return []

        try:
            # Ensure confidence column is numeric, coercing errors
            confidence_numeric = pd.to_numeric(
                self._df[CONFIDENCE_COL], errors="coerce"
            )
            mask = confidence_numeric < threshold
            if not mask.any():
                return []

            # Get the index of the matching rows
            matching_index = self._df[mask].index

            # Extract unique primary entity IDs
            if isinstance(matching_index, pd.MultiIndex):
                if self._identifier_column not in matching_index.names:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' not found in MultiIndex names: {matching_index.names}. Cannot get low confidence entities."
                    )
                    return []
                unique_entity_ids = (
                    matching_index.get_level_values(self._identifier_column)
                    .unique()
                    .tolist()
                )
            else:  # Single index
                if self._identifier_column != matching_index.name:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' does not match single index name '{matching_index.name}'. Cannot get low confidence entities."
                    )
                    return []
                unique_entity_ids = matching_index.unique().tolist()

            return [str(eid) for eid in unique_entity_ids]  # Ensure IDs are strings
        except Exception as e:
            logger.error(f"Error in {operation}: {e}", exc_info=True)
            # Avoid raising? Log and return empty list.
            return []

    def find_entities_lacking_attributes(
        self, attributes: Optional[List[str]] = None
    ) -> List[str]:
        """
        Finds primary entity IDs that have at least one row lacking one or more specified attributes.
        If attributes is None, checks against all user-defined columns (excluding index columns).
        """
        operation = "find_entities_lacking_attributes"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert (
            self._identifier_column is not None
        )  # Need identifier to return correct IDs

        if self._df.empty:
            return []

        check_cols = attributes if attributes else self._user_columns
        # Exclude index columns from the check, as they must be present
        check_cols = [
            col
            for col in check_cols
            if col in self._df.columns and col not in self._index_columns
        ]

        if not check_cols:
            logger.debug(
                "No valid non-index attributes specified or available to check for missing values."
            )
            return []

        try:
            # Check for nulls in any of the specified columns for each row
            missing_mask = self._df[check_cols].isnull().any(axis=1)

            if not missing_mask.any():
                return []  # No rows have missing attributes in checked columns

            # Get the index of rows with missing attributes
            incomplete_index = self._df[missing_mask].index

            # Extract unique primary entity IDs from the incomplete index
            if isinstance(incomplete_index, pd.MultiIndex):
                if self._identifier_column not in incomplete_index.names:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' not found in MultiIndex names: {incomplete_index.names}. Cannot find entities lacking attributes."
                    )
                    return []
                unique_entity_ids = (
                    incomplete_index.get_level_values(self._identifier_column)
                    .unique()
                    .tolist()
                )
            else:  # Single index
                if self._identifier_column != incomplete_index.name:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' does not match single index name '{incomplete_index.name}'. Cannot find entities lacking attributes."
                    )
                    return []
                unique_entity_ids = incomplete_index.unique().tolist()

            logger.debug(
                f"Found {len(unique_entity_ids)} entities with rows lacking one or more attributes: {check_cols}."
            )
            return [str(eid) for eid in unique_entity_ids]  # Ensure IDs are strings

        except KeyError as e:
            logger.error(
                f"Error in {operation}: One or more check columns not found: {e}",
                exc_info=False,
            )
            # Return empty list as we cannot perform the check
            return []
        except Exception as e:
            logger.error(f"Error in {operation}: {e}", exc_info=True)
            # Avoid raising? Log and return empty list.
            return []

    def get_all_entity_ids(self) -> List[str]:
        """Retrieves a list of all unique primary entity IDs."""
        operation = "get_all_entity_ids"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert (
            self._identifier_column is not None
        )  # Need identifier to know which level/index to check

        if self._df.empty:
            return []
        try:
            if isinstance(self._df.index, pd.MultiIndex):
                # Get unique values from the level corresponding to the identifier column
                if self._identifier_column not in self._df.index.names:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' not found in MultiIndex names: {self._df.index.names}. Cannot get all entity IDs."
                    )
                    return []
                identifier_level_name = self._identifier_column
                return (
                    self._df.index.get_level_values(identifier_level_name)
                    .unique()
                    .tolist()
                )
            else:  # Single index
                if self._identifier_column != self._df.index.name:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' does not match single index name '{self._df.index.name}'. Cannot get all entity IDs."
                    )
                    return []
                return self._df.index.unique().tolist()
        except Exception as e:
            logger.error(f"Error retrieving all entity IDs: {e}", exc_info=True)
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e

    def clear_obstacles_for_entity(self, entity_id: str):
        """Clears the obstacles list for all rows associated with a specific primary entity ID."""
        operation = "clear_obstacles_for_entity"
        self._raise_if_not_initialized(operation)
        assert self._df is not None
        assert self._identifier_column is not None  # Need identifier to select rows

        if OBSTACLES_COL not in self._df.columns:
            logger.warning(
                f"Obstacles column '{OBSTACLES_COL}' does not exist. Cannot clear."
            )
            return

        try:
            # Select rows matching the primary entity ID
            if isinstance(self._df.index, pd.MultiIndex):
                if self._identifier_column not in self._df.index.names:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' not found in MultiIndex names: {self._df.index.names}. Cannot clear obstacles."
                    )
                    return
                identifier_level_name = self._identifier_column

                if entity_id not in self._df.index.get_level_values(
                    identifier_level_name
                ):
                    logger.debug(
                        f"Entity '{entity_id}' not found. Cannot clear obstacles."
                    )
                    return
                entity_rows_mask = (
                    self._df.index.get_level_values(identifier_level_name) == entity_id
                )
            else:  # Single index
                if self._identifier_column != self._df.index.name:
                    logger.error(
                        f"Identifier column '{self._identifier_column}' does not match single index name '{self._df.index.name}'. Cannot clear obstacles."
                    )
                    return
                if entity_id not in self._df.index:
                    logger.debug(
                        f"Entity '{entity_id}' not found. Cannot clear obstacles."
                    )
                    return
                entity_rows_mask = self._df.index == entity_id

            # Set the obstacle column to None for these rows
            # Use loc with the boolean mask to modify in place
            self._df.loc[entity_rows_mask, OBSTACLES_COL] = None
            logger.info(f"Cleared obstacles for entity '{entity_id}'.")

        except KeyError:
            logger.debug(
                f"Entity '{entity_id}' not found (KeyError during obstacle clearing)."
            )
            # Don't raise, just log
        except Exception as e:
            logger.error(
                f"Error clearing obstacles for entity '{entity_id}': {e}", exc_info=True
            )
            op_data = OperationalErrorData(
                component="TabularStore",
                operation=operation,
                details=f"Entity ID '{entity_id}': {e}",
            )
            raise KBInteractionError(
                f"Failed operation '{operation}' for entity '{entity_id}': {e}",
                operation_data=op_data,
            ) from e

    def reset_storage(self):
        """Resets the store to an uninitialized state."""
        operation = "reset_storage"
        logger.warning("Resetting TabularStore storage.")
        try:
            self._df = None
            self._task_definition = None
            self._identifier_column = None
            self._columns = []
            self._user_columns = []
            self._index_columns = []
            self._is_initialized = False
            logger.info("TabularStore storage reset successfully.")
        except Exception as e:
            logger.error(f"Error during storage reset: {e}", exc_info=True)
            # Ensure state reflects failure
            self._is_initialized = False
            self._df = None
            op_data = OperationalErrorData(
                component="TabularStore", operation=operation, details=str(e)
            )
            raise KBInteractionError(
                f"Failed operation '{operation}': {e}", operation_data=op_data
            ) from e
