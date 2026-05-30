"""
agents/base_agent.py
Base agent interface for all game agents.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Common interface for all agents."""

    @property
    @abstractmethod
    def name(self):
        """Agent name for identification."""
        pass

    @abstractmethod
    def reset(self):
        """Reset agent state."""
        pass

    @abstractmethod
    def select_action(self, game, training=False):
        """Select action given game state."""
        pass