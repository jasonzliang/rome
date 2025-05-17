from abc import ABC, abstractmethod
import glob
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from .logger import get_logger
from .config import set_attributes_from_config


class Action(ABC):
    """Abstract base class for all actions"""

    def __init__(self, config: Dict = None):
        """Initialize the action with a configuration dictionary"""
        self.config = config or {}
        self.logger = get_logger()

        # Automatically set attributes from config
        set_attributes_from_config(self, self.config)

    @abstractmethod
    def execute(self, agent, **kwargs):
        """Execute the action and return the next state"""
        pass
