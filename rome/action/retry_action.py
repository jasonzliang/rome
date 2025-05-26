import glob
import os
import sys
import traceback
from typing import Dict, List, Any, Optional
from .action import Action
from ..logger import get_logger


class RetryAction(Action):
    """Action to retry, go back to idle state"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = get_logger()

        # No specific config requirements for RetryAction, but we should still initialize
        # with the config dict for consistency

    def summary(self, agent) -> str:
        """Return a short summary of the retry action"""
        return "Clear current context and return to idle state to start fresh"

    def execute(self, agent, **kwargs) -> bool:
        self.logger.info("Starting RetryAction execution")
        agent.context.clear()
        self.logger.info("Agent context has been cleared")
        return True
