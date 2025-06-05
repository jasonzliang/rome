# action_selector.py - Enhanced with backoff decorator
import time
import functools
import backoff
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from .logger import get_logger
from .config import set_attributes_from_config, check_attrs


class ActionSelectorError(Exception):
    """Custom exception for action selection errors"""
    pass


class ActionTooFrequentError(Exception):
    """Exception raised when max backoff attempts exceeded"""
    def __init__(self, action_name: str, attempts: int, max_tries: int):
        self.action_name = action_name
        self.attempts = attempts
        self.max_tries = max_tries
        super().__init__(f"Action '{action_name}' blocked after {attempts} backoff attempts (max: {max_tries})")


def action_frequency_backoff(func=None):
    """
    Compact backoff: doubles wait time on retry, raises exception after max_tries
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, agent, iteration: int, stop_on_error: bool = True):
            # Initialize tracking
            if not hasattr(self, 'last_execution'):
                self.last_execution = {}
            if not hasattr(self, 'skip_next_check'):
                self.skip_next_check = set()
            if not hasattr(self, 'backoff_attempts'):
                self.backoff_attempts = {}

            if not getattr(self, 'backoff_enabled', True):
                return f(self, agent, iteration, stop_on_error)

            # Get action result
            result = f(self, agent, iteration, stop_on_error)
            if not result or len(result) != 3:
                return result

            action_name, reason, should_continue = result
            if not action_name or not should_continue:
                return result

            # Skip check if action just completed backoff
            if action_name in self.skip_next_check:
                self.skip_next_check.remove(action_name)
                self.last_execution[action_name] = time.time()
                self.logger.debug(f"Skipping frequency check for {action_name} (post-backoff)")
                return result

            # Check frequency
            now = time.time()
            last_time = self.last_execution.get(action_name, 0)
            elapsed = now - last_time

            if elapsed < self.min_interval:
                # Track backoff attempts
                attempts = self.backoff_attempts.get(action_name, 0) + 1
                self.backoff_attempts[action_name] = attempts

                # Check if max tries exceeded
                if attempts > self.max_tries:
                    self.logger.error(f"Action {action_name} blocked after {attempts} attempts")
                    raise ActionTooFrequentError(action_name, attempts, self.max_tries)

                # Calculate exponentially increasing wait time: min_interval * 2^attempts
                wait_time = self.min_interval * (2 ** attempts)

                self.logger.debug(f"Action {action_name} backing off {wait_time:.1f}s "
                                f"(attempt {attempts}/{self.max_tries})")

                # Sleep and mark to skip next check
                time.sleep(wait_time)
                self.skip_next_check.add(action_name)
            else:
                # Reset attempts counter on successful execution within interval
                self.backoff_attempts[action_name] = 0

            # Record execution time
            self.last_execution[action_name] = time.time()
            return result

        return wrapper

    return decorator if func is None else decorator(func)


class ActionSelectorBase(ABC):
    """Abstract base class for action selection strategies"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()
        set_attributes_from_config(self, self.config)

    @abstractmethod
    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Generate action selection prompt based on current FSM state and agent context"""
        pass

    def _extract_action_from_response(self, agent, response: str) -> tuple:
        """Extract the action name from the LLM response"""
        reasoning = "No reasoning provided"

        # Try to parse as JSON first
        parsed_json = agent.parse_json_response(response)

        if parsed_json and 'selected_action' in parsed_json:
            action = parsed_json['selected_action']
            reasoning = parsed_json.get('reasoning', 'No reasoning provided')
            self.logger.info(f"Selected action: {action} - {reasoning}")
            return action, reasoning

        # Fallback: look for action names in the response text
        available_actions = agent.fsm.get_available_actions()
        for action in available_actions:
            if action.lower() in response.lower():
                self.logger.info(f"Extracted action from text: {action}")
                return action, reasoning

        # If no action found, log the issue and return None
        self.logger.error(f"Could not extract valid action from response: {response}")
        raise ValueError(f"Could not extract valid action from response: {response}")

    def select_action(self, agent, iteration: int, stop_on_error: bool = True) -> Tuple[str, str, bool]:
        """Select an action with frequency-based exponential backoff"""
        # Check if there are available actions
        available_actions = agent.fsm.get_available_actions()
        if not available_actions:
            error_msg = "No available actions in current state"
            self.logger.error(f"{error_msg}. Stopping loop.")

            if stop_on_error:
                raise ActionSelectorError(error_msg)
            else:
                agent.fsm.reset(agent)
                agent.history.reset()
                return None, f"{error_msg} - reset and continue", True
        else:
            self.logger.info(f"Available actions from {agent.fsm.get_current_state()}: {available_actions}")

        # If only one action available, select it directly without LLM call
        if len(available_actions) == 1:
            chosen_action = available_actions[0]
            reasoning = "only one action available - auto-selected"
            self.logger.info(f"Auto-selecting single available action: {chosen_action}")
            return chosen_action, reasoning, True

        # Multiple actions available - use LLM for selection
        try:
            # Use strategy-specific prompt generation
            prompt = self.get_action_selection_prompt(agent.fsm, agent)

            # Get action choice from LLM
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role,
                response_format={"type": "json_object"}
            )

            # Extract action from response
            chosen_action, reasoning = self._extract_action_from_response(agent, response)

        except Exception as e:
            error_msg = f"LLM call failed during action selection: {str(e)}"
            self.logger.error(error_msg)
            agent.history.add_error(iteration, error_msg, agent.fsm.current_state, str(e))

            if stop_on_error:
                raise ActionSelectorError(error_msg) from e
            else:
                agent.fsm.reset(agent)
                agent.history.reset()
                return None, f"{error_msg} - reset and continue", True

        # Validate action was extracted
        if not chosen_action:
            error_msg = f"Could not determine action from LLM response: {response}"
            self.logger.error(error_msg)
            agent.history.add_error(iteration, error_msg, agent.fsm.current_state)

            if stop_on_error:
                raise ActionSelectorError(error_msg)
            else:
                agent.fsm.reset(agent)
                agent.history.reset()
                return None, f"{error_msg} - reset and continue", True

        # Validate action is available
        if chosen_action not in available_actions:
            error_msg = f"Action '{chosen_action}' not available in state '{agent.fsm.current_state}'. Available: {available_actions}"
            self.logger.error(error_msg)
            agent.history.add_error(iteration, error_msg, agent.fsm.current_state)

            if stop_on_error:
                raise ActionSelectorError(error_msg)
            else:
                agent.fsm.reset(agent)
                agent.history.reset()
                return None, f"{error_msg} - reset and continue", True

        return chosen_action, reasoning, True

    # def get_backoff_status(self) -> Dict:
    #     """Get current backoff status"""
    #     if hasattr(self, 'last_execution'):
    #         current_time = time.time()
    #         return {
    #             action: {
    #                 'time_since_last': current_time - last_exec,
    #                 'can_execute': (current_time - last_exec) >= self.min_interval
    #             }
    #             for action, last_exec in self.last_execution.items()
    #         }
    #     return {}

    # def reset_backoff(self, action_name: str = None):
    #     """Reset backoff tracking"""
    #     if hasattr(self, 'last_execution'):
    #         if action_name:
    #             self.last_execution.pop(action_name, None)
    #             self.logger.info(f"Reset backoff for '{action_name}'")
    #         else:
    #             self.last_execution.clear()
    #             self.logger.info("Reset all backoff tracking")


class OriginalActionSelector(ActionSelectorBase):
    """Original action selection method - preserves existing behavior"""

    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Original action selection prompt logic"""
        if fsm.current_state not in fsm.states:
            raise ValueError(f"Current state '{fsm.current_state}' is not a valid state in the FSM")

        state_summary = f"{fsm.current_state}: {fsm.states[fsm.current_state].summary(agent)}"
        available_actions = fsm.get_available_actions()

        # Build action list with summaries and transition information
        action_details = []
        future_states = set()  # Track unique future states for summaries

        for action_name in available_actions:
            assert action_name in fsm.actions
            action_summary = fsm.actions[action_name].summary(agent)

            # Get transition information
            assert fsm.current_state in fsm.transitions and action_name in fsm.transitions[fsm.current_state]
            target_state, fallback_state = fsm.transitions[fsm.current_state][action_name]
            transition_info = f"next state: {target_state}"
            if fallback_state:
                transition_info += f", fallback: {fallback_state}"
                future_states.add(fallback_state)
            future_states.add(target_state)

            action_details.append(f"- {action_name} ({transition_info}): {action_summary}")

        actions_text = "\n".join(action_details)

        # Build future states summaries, sort for consistent ordering
        future_summaries = []
        for state_name in sorted(future_states):
            assert state_name in fsm.states
            future_summary = fsm.states[state_name].future_summary(agent)
            future_summaries.append(f"- {state_name}: {future_summary}")

        future_states_text = "\n".join(future_summaries) if future_summaries else "No future states to display"

        # Get history summary based on config
        history_summary = agent.history.get_history_summary(agent.history_context_len)

        prompt_parts = []

        # Add overview if it's set
        if fsm.overview:
            prompt_parts.append(f"## FSM Overview ##\n{fsm.overview}")

        # Add other sections
        if history_summary:
            prompt_parts.append(f"## Recent agent history ##\n{history_summary}")

        prompt_parts.extend([
            f"## Current state ##\n{state_summary}",
            f"## Available actions ##\n{actions_text}",
            f"## Future states ##\n{future_states_text}"
        ])

        prompt_parts.append(
"""Please select one of the available actions to execute. Respond with a JSON object containing:
{
    "initial_action": "initial considered action name",
    "selected_action": "final selected action name",
    "reasoning": "Short reason of why you choose this final action and why it is different from initial action"
}"""
        )

        prompt_parts.append(f"IMPORTANT:\nChoose the most appropriate action using your role as a guide. First consider an initial action and if that action (and associated state) is getting repeated {agent.patience} times or more in recent history, try selecting an alternative action to avoid getting stuck. Diversity in action selection often leads to better outcomes. Please mention in reasoning if avoiding being stuck influenced the final action that was selected.")

        return "\n\n".join(prompt_parts)


class SmartActionSelector(ActionSelectorBase):
    """Enhanced action selection with intelligent loop detection and patience-based intervention"""

    def __init__(self, config: Dict = None):
        """Initialize SmartActionSelector with configurable parameters"""
        super().__init__(config)

        check_attrs(self, [
            'loop_detection_window', # None = use agent.history_context_len
            'exploration_rate', # Base exploration probability
            'overuse_threshold' # Fraction triggering overuse warning
        ])

        self.logger.info(f"SmartActionSelector initialized with config: {self.config}")

    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Enhanced action selection prompt with simplified anti-loop logic"""
        if fsm.current_state not in fsm.states:
            raise ValueError(f"Current state '{fsm.current_state}' is not a valid state in the FSM")

        # Build base prompt components
        prompt_parts = self._build_base_prompt_parts(fsm, agent)

        # Analyze patterns and add intervention if needed
        pattern_analysis = self._analyze_action_patterns(agent, fsm.get_available_actions(), fsm.current_state)

        # Add selection instruction with dynamic guidance
        selection_instruction = self._build_selection_instruction(pattern_analysis, agent)
        prompt_parts.append(selection_instruction)

        return "\n\n".join(prompt_parts)

    def _build_base_prompt_parts(self, fsm, agent) -> List[str]:
        """Build the standard prompt components"""
        state_summary = f"{fsm.current_state}: {fsm.states[fsm.current_state].summary(agent)}"
        available_actions = fsm.get_available_actions()

        # Build action details
        action_details = []
        future_states = set()

        for action_name in available_actions:
            action_summary = fsm.actions[action_name].summary(agent)
            target_state, fallback_state = fsm.transitions[fsm.current_state][action_name]

            transition_info = f"next state: {target_state}"
            if fallback_state:
                transition_info += f", fallback: {fallback_state}"
                future_states.add(fallback_state)
            future_states.add(target_state)

            action_details.append(f"- {action_name} ({transition_info}): {action_summary}")

        actions_text = "\n".join(action_details)

        # Build future states summaries
        future_summaries = []
        for state_name in sorted(future_states):
            future_summary = fsm.states[state_name].future_summary(agent)
            future_summaries.append(f"- {state_name}: {future_summary}")

        future_states_text = "\n".join(future_summaries) if future_summaries else "No future states to display"

        # Get history summary
        history_summary = agent.history.get_history_summary(agent.history_context_len)

        # Assemble prompt parts
        prompt_parts = []

        if fsm.overview:
            prompt_parts.append(f"## FSM Overview ##\n{fsm.overview}")

        if history_summary:
            prompt_parts.append(f"## Recent agent history ##\n{history_summary}")

        prompt_parts.extend([
            f"## Current state ##\n{state_summary}",
            f"## Available actions ##\n{actions_text}",
            f"## Future states ##\n{future_states_text}"
        ])

        return prompt_parts

    def _analyze_action_patterns(self, agent, available_actions: List[str], current_state: str) -> Dict:
        """Simplified pattern analysis focusing on patience-based intervention"""

        # Use agent's history context length or configured window
        window_size = self.loop_detection_window or agent.history_context_len
        recent_actions = agent.history.get_recent_actions(window_size)

        if len(recent_actions) < 3:
            return {"needs_intervention": False, "message": "", "severity": "none"}

        # Count state-action combinations in recent history
        state_action_counts = {}
        action_counts = {}

        for action_record in recent_actions:
            # Handle different possible record formats
            if isinstance(action_record, dict):
                action = action_record.get('action')
                state = action_record.get('prev_state') or action_record.get('state')
            else:
                action = action_record
                state = current_state  # Fallback assumption

            if action:
                action_counts[action] = action_counts.get(action, 0) + 1

                if state:
                    key = f"{state}->{action}"
                    state_action_counts[key] = state_action_counts.get(key, 0) + 1

        # Check for patience-based intervention (primary trigger)
        current_state_actions = [f"{current_state}->{action}" for action in available_actions]
        max_repetitions = max(
            (state_action_counts.get(key, 0) for key in current_state_actions),
            default=0
        )

        # Determine intervention level based on agent.patience
        patience_threshold = agent.patience
        critical_threshold = patience_threshold + 2  # Allow some buffer beyond patience

        severity = "none"
        needs_intervention = False
        message = ""

        if max_repetitions >= critical_threshold:
            severity = "critical"
            needs_intervention = True
            # Find the overused actions
            overused = [
                action for action in available_actions
                if state_action_counts.get(f"{current_state}->{action}", 0) >= critical_threshold
            ]
            underused = [
                action for action in available_actions
                if state_action_counts.get(f"{current_state}->{action}", 0) <= 1 and action not in overused
            ]

            message = f"🚨 CRITICAL: Repeated pattern detected!\n"
            message += f"- Actions used {max_repetitions}+ times: {', '.join(overused)}\n"
            message += f"- Consider alternatives: {', '.join(underused) if underused else 'any other action'}\n"
            message += f"- Patience threshold ({patience_threshold}) exceeded by {max_repetitions - patience_threshold}"

        elif max_repetitions >= patience_threshold:
            severity = "warning"
            needs_intervention = True
            # Identify specific overused actions
            overused = [
                action for action in available_actions
                if state_action_counts.get(f"{current_state}->{action}", 0) >= patience_threshold
            ]
            alternatives = [
                action for action in available_actions
                if state_action_counts.get(f"{current_state}->{action}", 0) < patience_threshold
            ]

            message = f"⚠️ WARNING: Approaching repetitive behavior\n"
            message += f"- Actions at patience limit ({patience_threshold}): {', '.join(overused)}\n"
            message += f"- Available alternatives: {', '.join(alternatives) if alternatives else 'none'}\n"
            message += f"- Consider diversifying approach"

        # Secondary check: overall action overuse in window
        elif len(recent_actions) >= 5:  # Only check if sufficient history
            total_actions = len(recent_actions)
            overuse_threshold = max(2, int(total_actions * self.overuse_threshold))

            heavily_used = [
                action for action, count in action_counts.items()
                if count >= overuse_threshold and action in available_actions
            ]

            if heavily_used:
                severity = "mild"
                needs_intervention = True
                usage_pct = max(action_counts.get(action, 0) for action in heavily_used) / total_actions * 100

                message = f"💡 INFO: Some actions heavily used in recent history\n"
                message += f"- Frequently used: {', '.join(heavily_used)} ({usage_pct:.0f}% of recent actions)\n"
                message += f"- Consider occasional exploration for better outcomes"

        return {
            "needs_intervention": needs_intervention,
            "message": message,
            "severity": severity,
            "max_repetitions": max_repetitions,
            "patience_threshold": patience_threshold
        }

    def _build_selection_instruction(self, pattern_analysis: Dict, agent) -> str:
        """Build selection instruction with appropriate guidance level"""

        base_instruction = """Please select one of the available actions to execute. Respond with a JSON object containing:
{
    "selected_action": "selected action name",
    "reasoning": "Short reason of why you choose this action"
}
"""

        if not pattern_analysis["needs_intervention"]:
            # Minimal guidance when no patterns detected
            guidance = f"\n\nIMPORTANT:\nChoose the most appropriate action for your role. "
            if self.exploration_rate > 0.1:
                guidance += f"Occasional exploration can lead to better outcomes."
            return base_instruction + guidance

        # Pattern-specific guidance based on severity
        severity = pattern_analysis["severity"]

        if severity == "critical":
            guidance = f"\n\nCRITICAL INTERVENTION NEEDED:\n{pattern_analysis['message']}\n"
            guidance += f"STRONGLY consider selecting a different action to break the repetitive cycle. "
            guidance += f"The current approach has exceeded patience limits and may be ineffective."

        elif severity == "warning":
            guidance = f"\n\nPATIENCE THRESHOLD REACHED:\n{pattern_analysis['message']}\n"
            guidance += f"Consider whether the current approach is making progress. "
            guidance += f"If not, try a different action to avoid getting stuck."

        elif severity == "mild":
            guidance = f"\n\nEXPLORATION SUGGESTION:\n{pattern_analysis['message']}\n"
            guidance += f"Current strategy may be working, but diversity often improves outcomes."

        return base_instruction + guidance

    def _get_adaptive_window_size(self, agent) -> int:
        """Get analysis window size, defaulting to agent's history context length"""
        return self.loop_detection_window or agent.history_context_len


class ActionSelector:
    """Factory class for action selection strategies with backoff support"""
    SELECTORS = {
        "original": OriginalActionSelector,
        "smart": SmartActionSelector
    }

    def __init__(self, strategy: str = "original", config: Dict = None):
        """Initialize action selection with specified strategy"""
        self.strategy = strategy
        self.config = config or {}
        self.logger = get_logger()

        set_attributes_from_config(self, self.config, ['min_interval', 'max_tries', 'backoff_enabled'])

        # Run selector constructor
        if strategy not in self.SELECTORS:
            raise ValueError(f"Unknown action selection strategy: {strategy}. Available: {list(self.SELECTORS.keys())}")

        strategy_config = self.config.get(strategy, {})
        self.selector = self.SELECTORS[self.strategy](strategy_config)

        self.logger.info(f"Initialized {strategy} action selection strategy with backoff "
            f"enabled: {self.backoff_enabled}")

    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Generate action selection prompt using the configured strategy"""
        return self.selector.get_action_selection_prompt(fsm, agent)

    @action_frequency_backoff()
    def select_action(self, agent, iteration: int, stop_on_error: bool = True) -> Tuple[str, str, bool]:
        """Select an action using the configured strategy with automatic backoff"""
        return self.selector.select_action(agent, iteration, stop_on_error)
