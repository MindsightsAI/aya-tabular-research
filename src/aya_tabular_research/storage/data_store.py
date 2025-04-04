from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd

from ..core.models.research_models import TaskDefinition

class DataStore(ABC):
    """
    Abstract Base Class defining the interface for storing and managing
    the structured research data (e.g., entities and their attributes).
    """

    @abstractmethod
    def initialize_storage(self, task_definition: TaskDefinition):
        """
        Initializes the data storage based on the provided task definition.
        This typically involves setting up the schema (e.g., columns).

        Args:
            task_definition: The definition of the research task.
        """
        pass

    @abstractmethod
    def add_or_update_entity(self, entity_id: str, attributes: Dict[str, Any]):
        """
        Adds a new entity or updates an existing entity's attributes.

        Args:
            entity_id: The unique identifier of the entity.
            attributes: A dictionary of attribute names and their values to update.
                        The entity_id itself might be included here or handled separately.
        """
        pass

    @abstractmethod
    def get_entity_data(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves all known attributes for a specific entity.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            A dictionary of the entity's attributes, or None if the entity is not found.
        """
        pass

    @abstractmethod
    def get_all_data(self) -> pd.DataFrame:
        """
        Retrieves all data currently stored.

        Returns:
            A pandas DataFrame containing all entities and their attributes.
        """
        pass

    @abstractmethod
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Provides a summary of the current data state (e.g., number of entities,
        completeness statistics).

        Returns:
            A dictionary containing summary information.
        """
        pass

    @abstractmethod
    def find_entities_lacking_attributes(self, required_attributes: List[str]) -> List[str]:
        """
        Identifies entities that are missing one or more specified required attributes.

        Args:
            required_attributes: A list of attribute names that are considered required.

        Returns:
            A list of entity IDs that are missing at least one required attribute.
        """
        pass

    # Persistence methods removed for stateless operation
    # @abstractmethod
    # def save_data(self, file_path: str): ...
    # @abstractmethod
    # def load_data(self, file_path: str): ...

    @abstractmethod
    def reset_storage(self):
        """
        Resets the data store to an empty state, potentially keeping the schema.
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Returns True if the storage has been initialized, False otherwise."""
        pass

    @property
    @abstractmethod
    def identifier_column(self) -> Optional[str]:
        """Returns the name of the identifier column."""
        pass

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """Returns the list of all column names."""
        pass