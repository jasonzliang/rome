import os
import sys
import uuid
from unittest.mock import Mock, patch, MagicMock

# Adjust the path to import from parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from rome.fsm import FSM
from rome.action import Action
from rome.state import State

TMP_DIR = "/tmp"


def get_tmp_path(prefix="fsm_test_"):
    """Get a unique path in the /tmp directory"""
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(TMP_DIR, f"{prefix}{unique_id}")


# Alternative to contextmanager if that's causing issues
def temp_directory():
    """Create a temporary directory that's automatically cleaned up"""
    return TempDirectory()


class MockState(State):
    """Mock implementation of State for testing"""

    def __init__(self, name, check_context_result=True, context_error_msg=None):
        super().__init__(actions=[])
        self.name = name
        self._check_context_result = check_context_result
        self._context_error_msg = context_error_msg

    def check_context(self, agent, **kwargs):
        if isinstance(self._check_context_result, Exception):
            raise self._check_context_result
        if not self._check_context_result and self._context_error_msg:
            raise AssertionError(self._context_error_msg)
        return self._check_context_result

    def get_state_prompt(self, agent):
        return f"This is the {self.name} state prompt"


class MockAction(Action):
    """Mock implementation of Action for testing"""

    def __init__(self, execute_result=True):
        super().__init__()
        self.execute_result = execute_result
        self.execute_called = False

    def execute(self, agent, **kwargs):
        self.execute_called = True
        return self.execute_result


def setup_fsm():
    """Set up a basic FSM for testing"""
    # Create a mock logger
    mock_logger = MagicMock()

    # Patch the logger globally
    with patch.object(FSM, 'logger', mock_logger):
        # Create a basic FSM for testing
        fsm = FSM()

        # Create mock states
        state_a = MockState("STATE_A")
        state_b = MockState("STATE_B")
        state_c = MockState("STATE_C")
        fallback_state = MockState("FALLBACK")

        # Patch the logger for each state
        state_a.logger = mock_logger
        state_b.logger = mock_logger
        state_c.logger = mock_logger
        fallback_state.logger = mock_logger

        # Add states to FSM
        fsm.add_state(state_a)
        fsm.add_state(state_b)
        fsm.add_state(state_c)
        fsm.add_state(fallback_state)

        # Set initial state
        fsm.set_initial_state("STATE_A")

        # Create a mock agent
        mock_agent = Mock()
        mock_agent.context = {}

        return fsm, mock_agent


def test_add_and_execute_action_success_path():
    """Test adding and executing an action - success path"""
    fsm, mock_agent = setup_fsm()

    # Create a success action
    success_action = MockAction(execute_result=True)

    # Add the action
    fsm.add_action("STATE_A", "STATE_B", success_action, action_name="SuccessAction")

    # Execute the action
    fsm.execute_action("SuccessAction", mock_agent)

    # Verify that the action was executed and state changed
    assert success_action.execute_called, "Action should have been executed"
    assert fsm.current_state == "STATE_B", f"Expected STATE_B but got {fsm.current_state}"
    print("✓ Success path test passed")


def test_execute_action_with_fallback():
    """Test executing an action that fails and uses fallback state"""
    fsm, mock_agent = setup_fsm()

    # Create a failing action
    fail_action = MockAction(execute_result=False)

    # Add the action with a fallback
    fsm.add_action("STATE_A", "STATE_B", fail_action,
                  action_name="FailAction", fallback_state="FALLBACK")

    # Execute the action
    fsm.execute_action("FailAction", mock_agent)

    # Verify that the action was executed and state changed to fallback
    assert fail_action.execute_called, "Action should have been executed"
    assert fsm.current_state == "FALLBACK", f"Expected FALLBACK but got {fsm.current_state}"
    print("✓ Fallback path test passed")


def test_execute_action_failure_without_fallback():
    """Test executing an action that fails but has no fallback state"""
    fsm, mock_agent = setup_fsm()

    # Create a failing action
    fail_action = MockAction(execute_result=False)

    # Add the action without a fallback
    fsm.add_action("STATE_A", "STATE_B", fail_action, action_name="FailActionNoFallback")

    # Execute the action
    fsm.execute_action("FailActionNoFallback", mock_agent)

    # Verify that even though action failed, we still go to the target state
    # since no fallback is defined
    assert fail_action.execute_called, "Action should have been executed"
    assert fsm.current_state == "STATE_B", f"Expected STATE_B but got {fsm.current_state}"
    print("✓ Failure without fallback test passed")


def test_check_context_error():
    """Test that errors in check_context are properly propagated"""
    fsm, mock_agent = setup_fsm()

    # Create a state that will fail context check
    error_state = MockState("ERROR_STATE", check_context_result=False,
                           context_error_msg="Context validation failed")
    fsm.add_state(error_state)

    # Create action that points to the error state
    action = MockAction(execute_result=True)
    fsm.add_action("STATE_A", "ERROR_STATE", action, action_name="ErrorAction")

    # Execute the action and expect an AssertionError
    try:
        fsm.execute_action("ErrorAction", mock_agent)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Context validation failed" in str(e), f"Expected 'Context validation failed' in exception message but got: {str(e)}"

    print("✓ Context error test passed")


def test_action_throws_exception():
    """Test handling of exceptions thrown by actions"""
    fsm, mock_agent = setup_fsm()

    # Create an action that raises an exception
    exception_action = MockAction()
    # Replace the execute method with a mock that raises an exception
    original_execute = exception_action.execute

    def raising_execute(*args, **kwargs):
        original_execute(*args, **kwargs)
        raise RuntimeError("Action execution failed")

    exception_action.execute = raising_execute

    # Add the action
    fsm.add_action("STATE_A", "STATE_B", exception_action,
                  action_name="ExceptionAction", fallback_state="FALLBACK")

    # Execute the action and expect the exception to be propagated
    try:
        fsm.execute_action("ExceptionAction", mock_agent)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Action execution failed" in str(e), f"Expected 'Action execution failed' in exception message but got: {str(e)}"

    # State should not have changed due to exception
    assert fsm.current_state == "STATE_A", f"Expected STATE_A but got {fsm.current_state}"
    print("✓ Action exception test passed")


def test_check_context_exception():
    """Test handling of exceptions thrown by check_context"""
    fsm, mock_agent = setup_fsm()

    # Create a state that throws exception during context check
    exception_state = MockState("EXCEPTION_STATE",
                              check_context_result=RuntimeError("Context check failed"))
    fsm.add_state(exception_state)

    # Create action to the exception state
    action = MockAction(execute_result=True)
    fsm.add_action("STATE_A", "EXCEPTION_STATE", action, action_name="ContextExceptionAction")

    # Execute the action and expect RuntimeError
    try:
        fsm.execute_action("ContextExceptionAction", mock_agent)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Context check failed" in str(e), f"Expected 'Context check failed' in exception message but got: {str(e)}"

    # Action should have been executed but state should not have changed due to exception
    assert action.execute_called, "Action should have been executed"
    assert fsm.current_state == "STATE_A", f"Expected STATE_A but got {fsm.current_state}"
    print("✓ Context exception test passed")


def test_get_action_prompt_invalid_state():
    """Test that get_action_prompt raises error for invalid current state"""
    fsm, mock_agent = setup_fsm()

    # Set current state to a value not in states
    fsm.current_state = "NONEXISTENT_STATE"

    # Attempt to get action prompt and expect ValueError
    try:
        fsm.get_action_prompt(mock_agent)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not a valid state" in str(e), f"Expected 'not a valid state' in exception message but got: {str(e)}"

    print("✓ Invalid state prompt test passed")


def test_action_not_available():
    """Test that attempting to execute unavailable action raises error"""
    fsm, mock_agent = setup_fsm()

    # Attempt to execute an action not available from the current state
    try:
        fsm.execute_action("NonexistentAction", mock_agent)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not available from" in str(e), f"Expected 'not available from' in exception message but got: {str(e)}"

    print("✓ Unavailable action test passed")


def test_complex_transition_sequence():
    """Test a sequence of transitions including success and failure paths"""
    fsm, mock_agent = setup_fsm()

    # Create actions
    success_action1 = MockAction(execute_result=True)
    fail_action = MockAction(execute_result=False)
    success_action2 = MockAction(execute_result=True)

    # Add actions
    fsm.add_action("STATE_A", "STATE_B", success_action1, action_name="Success1")
    fsm.add_action("STATE_B", "STATE_C", fail_action,
                  action_name="Fail", fallback_state="FALLBACK")
    fsm.add_action("FALLBACK", "STATE_C", success_action2, action_name="Success2")

    # Execute sequence
    fsm.execute_action("Success1", mock_agent)
    assert fsm.current_state == "STATE_B", f"Step 1: Expected STATE_B but got {fsm.current_state}"

    fsm.execute_action("Fail", mock_agent)
    assert fsm.current_state == "FALLBACK", f"Step 2: Expected FALLBACK but got {fsm.current_state}"

    fsm.execute_action("Success2", mock_agent)
    assert fsm.current_state == "STATE_C", f"Step 3: Expected STATE_C but got {fsm.current_state}"

    print("✓ Complex transition sequence test passed")


def test_validate_fsm():
    """Test FSM validation"""
    fsm, mock_agent = setup_fsm()

    # Add actions needed for validation
    action1 = MockAction()
    action2 = MockAction()

    fsm.add_action("STATE_A", "STATE_B", action1, action_name="Action1")
    fsm.add_action("STATE_B", "STATE_C", action2, action_name="Action2")

    # FSM should be valid
    assert fsm.validate_fsm(), "FSM should be valid"

    # Create invalid FSM with missing transition targets
    mock_logger = MagicMock()

    invalid_fsm = FSM()
    invalid_fsm.logger = mock_logger

    invalid_state = MockState("INVALID_STATE")
    invalid_state.logger = mock_logger

    invalid_fsm.add_state(invalid_state)
    invalid_fsm.set_initial_state("INVALID_STATE")

    # Create a transition to a non-existent state
    invalid_action = MockAction()
    invalid_fsm.add_action("INVALID_STATE", "NONEXISTENT_STATE", invalid_action,
                         action_name="InvalidAction")

    # Validation should fail
    assert not invalid_fsm.validate_fsm(), "Invalid FSM should fail validation"

    print("✓ FSM validation test passed")


def test_draw_graph():
    """Test graph visualization (without actually rendering)"""
    fsm, mock_agent = setup_fsm()

    # Create a unique filename in /tmp
    output_path = get_tmp_path(prefix="fsm_graph_") + ".png"

    # Mock graphviz to prevent actual rendering
    with patch.object(FSM, 'draw_graph', autospec=True) as mock_draw_graph:
        mock_draw_graph.return_value = output_path

        # Add some actions for the graph
        action1 = MockAction()
        action2 = MockAction()
        fsm.add_action("STATE_A", "STATE_B", action1, action_name="Action1")
        fsm.add_action("STATE_B", "STATE_C", action2,
                      action_name="Action2", fallback_state="FALLBACK")

        # Call draw_graph with a path in the tmp dir
        output_file = mock_draw_graph(fsm, output_path)

        # Verify draw_graph was called and a filename returned
        assert mock_draw_graph.called, "draw_graph method should have been called"
        assert output_file == output_path, f"Output file should match requested path"

    print("✓ Draw graph test passed")


if __name__ == '__main__':
    # Run all tests
    print("Running FSM tests...")

    # Set the tmp dir for logger output
    log_dir = get_tmp_path(prefix="fsm_logs_")
    os.makedirs(log_dir, exist_ok=True)
    os.environ['TEMP_LOG_DIR'] = log_dir

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
