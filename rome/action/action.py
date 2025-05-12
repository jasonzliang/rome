# agent_actions.py
from abc import ABC, abstractmethod
import glob
import os
import sys
from typing import Dict, List, Any, Optional, Callable


class Action(ABC):
    """Abstract base class for all actions"""

    def __init__(self, config: Dict = None):
        """Initialize the action with a configuration dictionary"""
        self.config = config or {}

    @abstractmethod
    def execute(self, agent, **kwargs):
        """Execute the action and return the next state"""
        pass
