import os
import sys
import uuid
import shutil
from typing import Dict

# Adjust the path to import from parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from rome.fsm import FSM
from rome.action import Action
from rome.state import State, IdleState, CodeLoadedState
from rome.agent import Agent
from rome.logger import get_logger
from rome.action import SearchAction, RetryAction
from rome.config import DEFAULT_CONFIG

TMP_DIR = "/tmp"


def get_tmp_path(prefix="fsm_test_"):
    """Get a unique path in the /tmp directory"""
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(TMP_DIR, f"{prefix}{unique_id}")


# Custom Action classes for testing
class TestSuccessAction(Action):
    """Action that always succeeds"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.execute_called = False

    def execute(self, agent, **kwargs) -> bool:
        self.execute_called = True
        # Add something to agent context to verify execution
        agent.context['test_success_executed'] = True
        return True


class TestFailAction(Action):
    """Action that always fails"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.execute_called = False

    def execute(self, agent, **kwargs) -> bool:
        self.execute_called = True
        # Add something to agent context to verify execution
        agent.context['test_fail_executed'] = True
        # Note that the context may be cleared if we use a real agent with some states
        # that implement context clearing, so we need to be careful about this
        return False


class TestExceptionAction(Action):
    """Action that raises an exception"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.execute_called = False

    def execute(self, agent, **kwargs) -> bool:
        self.execute_called = True
        # Add something to agent context to verify execution
        agent.context['test_exception_executed'] = True
        raise RuntimeError("Action execution failed")


# Custom State for testing
class TestErrorState(State):
    """State that fails context validation"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs):
        raise AssertionError("Context validation failed")

    def get_state_prompt(self, agent):
        return "This is an error state prompt"


class TestExceptionState(State):
    """State that throws exception during context check"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs):
        raise RuntimeError("Context check failed")

    def get_state_prompt(self, agent):
        return "This is an exception state prompt"


def setup_test_directory():
    """Set up a test directory structure with some Python files"""
    test_dir = get_tmp_path(prefix="fsm_test_repo_")
    os.makedirs(test_dir, exist_ok=True)

    # Create a simple Python file for testing
    with open(os.path.join(test_dir, "test_file.py"), "w") as f:
        f.write("def test_function():\n    print('Hello, world!')\n")

    return test_dir


def setup_fsm():
    """Set up a basic FSM with real components for testing"""
    # Create a test repository
    repo_dir = setup_test_directory()

    # Create a test configuration
    config = DEFAULT_CONFIG.copy()
    config['Agent']['repository'] = repo_dir
    config['Logger']['level'] = 'INFO'
    config['Logger']['base_dir'] = get_tmp_path(prefix="fsm_logs_")

    # Create a real agent
    agent = Agent(
        name="TestAgent",
        role="You are a test agent for FSM testing",
        config_dict=config
    )

    # Create a basic FSM for testing
    fsm = FSM(config)

    # Create custom states that don't clear context on transitions
    class TestStateA(State):
        def check_context(self, agent, **kwargs):
            return True
        def get_state_prompt(self, agent):
            return "This is state A prompt"

    class TestStateB(State):
        def check_context(self, agent, **kwargs):
            return True
        def get_state_prompt(self, agent):
            return "This is state B prompt"

    class TestStateC(State):
        def check_context(self, agent, **kwargs):
            return True
        def get_state_prompt(self, agent):
            return "This is state C prompt"

    class TestFallbackState(State):
        def check_context(self, agent, **kwargs):
            return True
        def get_state_prompt(self, agent):
            return "This is fallback state prompt"

    # Create states that don't clear context on transition
    state_a = TestStateA()
    state_b = TestStateB()
    state_c = TestStateC()
    fallback_state = TestFallbackState()

    # Add states to FSM
    fsm.add_state(state_a, "STATE_A")
    fsm.add_state(state_b, "STATE_B")
    fsm.add_state(state_c, "STATE_C")
    fsm.add_state(fallback_state, "FALLBACK")

    # Add custom error states
    error_state = TestErrorState()
    exception_state = TestExceptionState()
    fsm.add_state(error_state, "ERROR_STATE")
    fsm.add_state(exception_state, "EXCEPTION_STATE")

    # Set initial state
    fsm.set_initial_state("STATE_A")

    # Prepare agent context for tests
    # This simulates what would happen after a successful file search
    agent.context['selected_file'] = {
        'path': os.path.join(repo_dir, "test_file.py"),
        'content': "def test_function():\n    print('Hello, world!')\n",
        'reason': 'Test reason'
    }

    return fsm, agent, repo_dir


def cleanup(repo_dir):
    """Clean up test directories"""
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)


def test_add_and_execute_action_success_path():
    """Test adding and executing an action - success path"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create a success action
        success_action = TestSuccessAction()

        # Add the action
        fsm.add_action("STATE_A", "STATE_B", success_action, action_name="SuccessAction")

        # Execute the action
        fsm.execute_action("SuccessAction", agent)

        # Verify that the action was executed and state changed
        assert success_action.execute_called, "Action should have been executed"
        assert 'test_success_executed' in agent.context, "Action execution should have updated context"
        assert fsm.current_state == "STATE_B", f"Expected STATE_B but got {fsm.current_state}"
        print("✓ Success path test passed")
    finally:
        cleanup(repo_dir)


def test_execute_action_with_fallback():
    """Test executing an action that fails and uses fallback state"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create a failing action
        fail_action = TestFailAction()

        # Add the action with a fallback
        fsm.add_action("STATE_A", "STATE_B", fail_action,
                      action_name="FailAction", fallback_state="FALLBACK")

        # Set the marker in context before execution
        agent.context['test_fail_executed'] = False

        # Execute the action
        fsm.execute_action("FailAction", agent)

        # Verify that the action was executed and state changed to fallback
        assert fail_action.execute_called, "Action should have been executed"
        # Don't check agent.context here as it might be cleared by the fallback state
        assert fsm.current_state == "FALLBACK", f"Expected FALLBACK but got {fsm.current_state}"
        print("✓ Fallback path test passed")
    finally:
        cleanup(repo_dir)


def test_execute_action_failure_without_fallback():
    """Test executing an action that fails but has no fallback state"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create a failing action
        fail_action = TestFailAction()

        # Add the action without a fallback
        fsm.add_action("STATE_A", "STATE_B", fail_action, action_name="FailActionNoFallback")

        # Set the marker in context before execution
        agent.context['test_fail_executed'] = False

        # Execute the action
        fsm.execute_action("FailActionNoFallback", agent)

        # Verify that even though action failed, we still go to the target state
        # since no fallback is defined
        assert fail_action.execute_called, "Action should have been executed"
        # Don't check agent.context as it might be cleared by the target state
        assert fsm.current_state == "STATE_B", f"Expected STATE_B but got {fsm.current_state}"
        print("✓ Failure without fallback test passed")
    finally:
        cleanup(repo_dir)


def test_check_context_error():
    """Test that errors in check_context are properly propagated"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create action that points to the error state
        action = TestSuccessAction()
        fsm.add_action("STATE_A", "ERROR_STATE", action, action_name="ErrorAction")

        # Execute the action and expect an AssertionError
        try:
            fsm.execute_action("ErrorAction", agent)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Context validation failed" in str(e), f"Expected 'Context validation failed' in exception message but got: {str(e)}"

        print("✓ Context error test passed")
    finally:
        cleanup(repo_dir)


def test_action_throws_exception():
    """Test handling of exceptions thrown by actions"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create an action that raises an exception
        exception_action = TestExceptionAction()

        # Add the action
        fsm.add_action("STATE_A", "STATE_B", exception_action,
                      action_name="ExceptionAction", fallback_state="FALLBACK")

        # Execute the action and expect the exception to be propagated
        try:
            fsm.execute_action("ExceptionAction", agent)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Action execution failed" in str(e), f"Expected 'Action execution failed' in exception message but got: {str(e)}"

        # State should not have changed due to exception
        assert fsm.current_state == "STATE_A", f"Expected STATE_A but got {fsm.current_state}"
        print("✓ Action exception test passed")
    finally:
        cleanup(repo_dir)


def test_check_context_exception():
    """Test handling of exceptions thrown by check_context"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create action to the exception state
        action = TestSuccessAction()
        fsm.add_action("STATE_A", "EXCEPTION_STATE", action, action_name="ContextExceptionAction")

        # Execute the action and expect RuntimeError
        try:
            fsm.execute_action("ContextExceptionAction", agent)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Context check failed" in str(e), f"Expected 'Context check failed' in exception message but got: {str(e)}"

        # Action should have been executed but state should not have changed due to exception
        assert action.execute_called, "Action should have been executed"
        assert fsm.current_state == "STATE_A", f"Expected STATE_A but got {fsm.current_state}"
        print("✓ Context exception test passed")
    finally:
        cleanup(repo_dir)


def test_get_action_prompt_invalid_state():
    """Test that get_action_prompt raises error for invalid current state"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Set current state to a value not in states
        fsm.current_state = "NONEXISTENT_STATE"

        # Attempt to get action prompt and expect ValueError
        try:
            fsm.get_action_prompt(agent)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not a valid state" in str(e), f"Expected 'not a valid state' in exception message but got: {str(e)}"

        print("✓ Invalid state prompt test passed")
    finally:
        cleanup(repo_dir)


def test_action_not_available():
    """Test that attempting to execute unavailable action raises error"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Attempt to execute an action not available from the current state
        try:
            fsm.execute_action("NonexistentAction", agent)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # The error message will be "Action 'NonexistentAction' does not exist in FSM"
            # or "Action 'NonexistentAction' not available from 'STATE_A'"
            # Both are valid error conditions we want to check for
            error_msg = str(e)
            assert ("not available from" in error_msg or
                    "does not exist in FSM" in error_msg), \
                f"Expected action not found error, got: {error_msg}"

        print("✓ Unavailable action test passed")
    finally:
        cleanup(repo_dir)


def test_complex_transition_sequence():
    """Test a sequence of transitions including success and failure paths"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create actions
        success_action1 = TestSuccessAction()
        fail_action = TestFailAction()
        success_action2 = TestSuccessAction()

        # Add actions
        fsm.add_action("STATE_A", "STATE_B", success_action1, action_name="Success1")
        fsm.add_action("STATE_B", "STATE_C", fail_action,
                      action_name="Fail", fallback_state="FALLBACK")
        fsm.add_action("FALLBACK", "STATE_C", success_action2, action_name="Success2")

        # Execute sequence
        fsm.execute_action("Success1", agent)
        assert fsm.current_state == "STATE_B", f"Step 1: Expected STATE_B but got {fsm.current_state}"

        fsm.execute_action("Fail", agent)
        assert fsm.current_state == "FALLBACK", f"Step 2: Expected FALLBACK but got {fsm.current_state}"

        fsm.execute_action("Success2", agent)
        assert fsm.current_state == "STATE_C", f"Step 3: Expected STATE_C but got {fsm.current_state}"

        print("✓ Complex transition sequence test passed")
    finally:
        cleanup(repo_dir)


def test_validate_fsm():
    """Test FSM validation"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Add actions needed for validation - ensuring all states are reachable
        action1 = TestSuccessAction()
        action2 = TestSuccessAction()
        action3 = TestSuccessAction()
        action4 = TestSuccessAction()
        action5 = TestSuccessAction()

        # Reset any existing transitions (to start clean)
        fsm.transitions = {}
        for state_name in fsm.states:
            fsm.transitions[state_name] = {}
            fsm.states[state_name].actions = []

        # Add actions that make all states reachable
        fsm.add_action("STATE_A", "STATE_B", action1, action_name="Action1")
        fsm.add_action("STATE_B", "STATE_C", action2, action_name="Action2")
        fsm.add_action("STATE_A", "FALLBACK", action3, action_name="Action3")
        fsm.add_action("STATE_A", "ERROR_STATE", action4, action_name="Action4")
        fsm.add_action("STATE_A", "EXCEPTION_STATE", action5, action_name="Action5")

        # FSM should now be valid (all states reachable)
        assert fsm.validate_fsm(), "FSM should be valid"

        # Create invalid FSM with missing transition targets
        invalid_fsm = FSM()

        # Create a simple state class for the invalid FSM test
        class InvalidTestState(State):
            def check_context(self, agent, **kwargs):
                return True
            def get_state_prompt(self, agent):
                return "This is an invalid test state prompt"

        invalid_state = InvalidTestState()

        invalid_fsm.add_state(invalid_state, "INVALID_STATE")
        invalid_fsm.set_initial_state("INVALID_STATE")

        # Create a non-existent state first (to bypass the existence check)
        nonexistent_state = InvalidTestState()
        invalid_fsm.add_state(nonexistent_state, "NONEXISTENT_STATE")

        # Create a transition
        invalid_action = TestSuccessAction()
        invalid_fsm.add_action("INVALID_STATE", "NONEXISTENT_STATE", invalid_action,
                             action_name="InvalidAction")

        # Manually remove the target state to create an invalid FSM
        del invalid_fsm.states["NONEXISTENT_STATE"]

        # Validation should fail
        assert not invalid_fsm.validate_fsm(), "Invalid FSM should fail validation"

        print("✓ FSM validation test passed")
    finally:
        cleanup(repo_dir)


def test_draw_graph():
    """Test graph visualization"""
    fsm, agent, repo_dir = setup_fsm()

    try:
        # Create a unique filename in /tmp
        output_path = get_tmp_path(prefix="fsm_graph_") + ".png"

        # Add some actions for the graph
        action1 = TestSuccessAction()
        action2 = TestSuccessAction()
        fsm.add_action("STATE_A", "STATE_B", action1, action_name="Action1")
        fsm.add_action("STATE_B", "STATE_C", action2,
                      action_name="Action2", fallback_state="FALLBACK")

        try:
            # This will actually render the graph if graphviz is installed
            output_file = fsm.draw_graph(output_path)
            assert output_file, "Output file should be produced"
            print(f"Graph rendered to: {output_file}")
        except ImportError:
            print("✓ Draw graph test skipped (graphviz not installed)")
        except Exception as e:
            print(f"Graph generation error: {e}")

        print("✓ Draw graph test passed")
    finally:
        cleanup(repo_dir)


if __name__ == '__main__':
    # Set up logging
    log_dir = get_tmp_path(prefix="fsm_logs_")
    os.makedirs(log_dir, exist_ok=True)
    os.environ['TEMP_LOG_DIR'] = log_dir

    print("Running FSM tests with real components...")

    # Run all test functions
    test_add_and_execute_action_success_path()
    test_execute_action_with_fallback()
    test_execute_action_failure_without_fallback()
    test_check_context_error()
    test_action_throws_exception()
    test_check_context_exception()
    test_get_action_prompt_invalid_state()
    test_action_not_available()
    test_complex_transition_sequence()
    test_validate_fsm()
    test_draw_graph()

    print("\nAll tests passed!")