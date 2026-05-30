"""
I have implemented and reviewed this module structure.
"""


from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        ...


    def reset(self) -> None:


        pass

    @abstractmethod

    def select_action(self, game, training: bool = False):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game, training.
        """
        ...
