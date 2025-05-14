import glob
import os
import sys
import traceback
from typing import Dict, List, Any, Optional
from action import Action

class RetryAction(Action):
"""Action to retry, go back to idle state"""

    def __init__(self, config: Dict):
        super().__init__(config)

    def execute(self, agent, **kwargs):
        agent.context = {}
