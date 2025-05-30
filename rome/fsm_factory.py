# fsm.py
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable, Union, Tuple

from .action import *
from .state import *
from .fsm import FSM
from .logger import get_logger

def create_minimal_fsm(config):
    """Create minimal FSM just two states"""
    logger = get_logger()
    logger.info("Creating FSM")

    fsm = FSM(config.get("FSM", {}))

    # Set overview for the minimal FSM
    fsm.set_overview(
        "This is a minimal code analysis worflow with a simple 2-state workflow: "
        "1) Start in IDLE state → Search and load code files → Move to CODE_LOADED state "
        "2) From CODE_LOADED state → Reset back to IDLE to start over. "
        "The agent's purpose is to help users locate, examine, and understand code within a codebase through an iterative search-and-reset cycle."
    )

    # Create and add states
    idle_state = fsm.add_state(IdleState(config.get('IdleState', {})))
    code_loaded_state = fsm.add_state(CodeLoadedState(config.get('CodeLoadedState', {})))

    # Create actions and add them to FSM
    fsm.add_action(idle_state, code_loaded_state,
                 SearchAction(config.get('SearchAction', {})),
                 fallback_state=idle_state)

    fsm.add_action(code_loaded_state, idle_state,
                 ResetAction(config.get('ResetAction', {})))

    # Set initial state and validate
    fsm.set_initial_state(idle_state)
    fsm.validate_fsm()

    logger.info("FSM created successfully")
    return fsm


def create_simple_fsm(config):
    """Create simple FSM with code editing, writing tests, and executing tests"""
    logger = get_logger()
    logger.info("Creating FSM with code editing, test writing and execution capabilities")

    fsm = FSM(config.get("FSM", {}))

    # Set overview for the simple FSM
    fsm.set_overview(
        "This is a comprehensive code development workflow with 6-states: "
        "1) IDLE → Search/load code → CODE_LOADED "
        "2) CODE_LOADED → Edit code OR write tests → CODE_EDITED or TEST_EDITED "
        "3) CODE_EDITED → Write/edit tests → TEST_EDITED "
        "4) TEST_EDITED → Execute code with tests → CODE_EXECUTED_PASS (success) or CODE_EXECUTED_FAIL (failure) "
        "5) CODE_EXECUTED_PASS → Reset → IDLE (complete successful cycle) "
        "6) CODE_EXECUTED_FAIL → Transition back to CODE_LOADED (retry) OR Reset → IDLE (start over). "
        "The agent follows a complete development lifecycle: discover code → modify code → create tests → validate → iterate or complete."
    )

    # Create and add states
    idle_state = fsm.add_state(IdleState(config.get('IdleState', {})))
    code_loaded_state = fsm.add_state(CodeLoadedState(config.get('CodeLoadedState', {})))
    code_edited_state = fsm.add_state(CodeEditedState(config.get('CodeEditedState', {})))
    test_edited_state = fsm.add_state(TestEditedState(config.get('TestEditedState', {})))
    code_executed_pass_state = fsm.add_state(
        CodeExecutedPassState(config.get('CodeExecutedPassState', {})))
    code_executed_fail_state = fsm.add_state(
        CodeExecutedFailState(config.get('CodeExecutedFailState', {})))

    # Create actions with their respective configurations
    search_action = SearchAction(config.get('SearchAction', {}))
    reset_action = AdvancedResetAction(config.get('AdvancedResetAction', {}))
    edit_code_action = EditCodeAction(config.get('EditCodeAction', {}))
    edit_test_action = EditTestAction(config.get('EditTestAction', {}))
    execute_code_action = ExecuteCodeAction(config.get('ExecuteCodeAction', {}),
        config.get('Executor', {}))
    transition_action = TransitionAction(config.get('TransitionAction', {}))

    # Add transitions from Idle state
    fsm.add_action(idle_state, code_loaded_state,
        search_action, fallback_state=idle_state)

    # Add transitions from CodeLoaded state
    fsm.add_action(code_loaded_state, code_edited_state, edit_code_action,
        fallback_state=idle_state)
    fsm.add_action(code_loaded_state, test_edited_state, edit_test_action,
        fallback_state=idle_state)

    # Add transitions from CodeEdited state
    fsm.add_action(code_edited_state, test_edited_state, edit_test_action,
        fallback_state=idle_state)

    # Add transitions from TestEdited state - THIS IS THE KEY CHANGE
    fsm.add_action(test_edited_state, code_executed_pass_state, execute_code_action,
        fallback_state=code_executed_fail_state)

    # Add transitions from CodeExecutedPass state
    fsm.add_action(code_executed_pass_state, idle_state, reset_action)

    # Add transitions from CodeExecutedFail state
    fsm.add_action(code_executed_fail_state, code_loaded_state, transition_action)
    fsm.add_action(code_executed_fail_state, idle_state, reset_action)

    # Set initial state and validate
    fsm.set_initial_state(idle_state)
    fsm.validate_fsm()

    logger.info("FSM created successfully with code editing, test writing, and execution capabilities")
    return fsm


FSM_FACTORY = {
    'minimal': create_minimal_fsm,
    'simple': create_simple_fsm,
}
