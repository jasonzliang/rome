# action.py
from abc import ABC, abstractmethod
import glob
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from .logger import get_logger


class Action(ABC):
    """Abstract base class for all actions"""

    def __init__(self, config: Dict = None):
        """Initialize the action with a configuration dictionary"""
        self.config = config or {}
        self.logger = get_logger()

    @abstractmethod
    def execute(self, agent, **kwargs):
        """Execute the action and return the next state"""
        pass