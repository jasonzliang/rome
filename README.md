# Rome

<p align="center"><img src="https://image.cdn2.seaart.me/2024-09-10/crg9uvle878c73cansn0/af22c8e52bb422f09e6813670d8b8166_high.webp" width="40%" alt="Rome"></p>

Rome is a Python library for building intelligent agents with Finite State Machine (FSM) architecture. This library provides a structured way to create AI-powered agents that can analyze, edit, and test code with LLM assistance.

## Overview

Rome is designed around a Finite State Machine architecture that allows AI agents to navigate through different states while performing operations on code. The library includes components for:

- State management
- Code analysis and editing
- Test creation and execution
- Visualization of agent state
- Versioning of code changes

### FSM Workflow

Rome’s default FSM supports a comprehensive 6-state development lifecycle:

1. **IDLE → CODE_LOADED**: Load or search for code to work on.
2. **CODE_LOADED → CODE_EDITED or TEST_EDITED**: Modify code or write tests.
3. **CODE_EDITED → TEST_EDITED**: After editing code, proceed to write/edit tests.
4. **TEST_EDITED → CODE_EXECUTED_PASS / CODE_EXECUTED_FAIL**: Execute tests; transition based on pass/fail.
5. **CODE_EXECUTED_PASS → IDLE**: Successful cycle completed.
6. **CODE_EXECUTED_FAIL → CODE_LOADED or IDLE**: Retry or restart.

This FSM ensures a structured agent behavior from discovery through validation.

## How It Works

Rome takes in a Python code file and performs the following:

1. **Analyzes the code structure**: Identifies function stubs, incomplete logic, and poor practices.
2. **Creates LLM prompts**: Generates structured prompts for agents to improve code quality.
3. **Executes & Evaluates**: Optionally runs the code or tests, gathers output, and suggests fixes.
4. **Generates unit tests**: Focuses on complete coverage including edge cases and error handling.

## Key Components

### Core Components

- **Agent**: Main class that orchestrates the entire workflow
- **FSM (Finite State Machine)**: Controls state transitions and available actions
- **States**: Different operational states the agent can be in (Idle, CodeLoaded, etc.)
- **Actions**: Operations that can be performed by the agent (Search, Edit, Execute)
- **OpenAI Handler**: Interface for LLM interactions

### States

The library implements several states:

- `IdleState`: Initial state waiting for tasks
- `CodeLoadedState`: A file has been selected for processing
- `CodeEditedState`: Code has been modified
- `TestEditedState`: Tests have been created/edited
- `CodeExecutedState`: Code or tests have been executed

### Actions

Available actions include:

- `SearchAction`: Find code files to work with in a repository
- `EditCodeAction`: Modify code with AI assistance
- `EditTestAction`: Create or improve tests
- `ExecuteCodeAction`: Run code and tests
- `RetryAction`: Reset and start over

### Visualization

The library includes a Streamlit-based visualization tool (`web_app.py`) that provides:

- Real-time FSM state visualization
- Agent context information
- Interactive configuration
- Graph representation of the state machine

## Getting Started

### Prerequisites

- Linux or MacOS with Python 3.10+ installed
- API key compatible with OpenAI's chat completion API

### Installation

#### From Source (Development)

```bash
git clone https://github.com/yourusername/rome.git
cd rome
pip install -e .        # Basic install
pip install -e .[dev]   # With dev tools
```

#### From PyPI (Coming Soon)

```bash
pip install rome
```

### Environment Setup

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Configuration

Rome uses a YAML-based configuration system. You can create your own `config.yaml` or generate one:

```python
# Generate a default configuration file
python -c "from config import generate_default_config; generate_default_config()"
```

Key configuration options:

- OpenAI model and parameters
- Repository path
- FSM type (`minimal`, `simple`, or custom)
- Logging settings
- API server settings (`host`, `port`)
- Patience setting to avoid stuck loops

Example snippet from a config:

```yaml
Agent:
  repository: "./"
  fsm_type: "simple"
  agent_api: true
  history_context_len: 15
  patience: 3

AgentApi:
  host: "localhost"
  port: 8000
```

### Basic Usage

```python
from agent import Agent

agent = Agent(
    name="CodingAgent",
    role="You are an expert Python developer focused on code quality and testing",
    config_dict={
        "Agent": {
            "repository": "./my_project",
            "fsm_type": "simple"
        }
    }
)

agent.run_loop(max_iterations=10)
```

### Visualizing Agent State

```bash
streamlit run web_app/streamlit_app.py
```

## Advanced Features

### Code Versioning

Rome automatically versions code files as they are modified, storing metadata in a `.rome/` directory.

### Custom FSMs

Define your own FSM logic:

```python
def create_custom_fsm(config):
    # Custom FSM logic here...
    return fsm_instance

FSM_FACTORY["custom"] = create_custom_fsm
```

### Extending with New Actions

Create custom actions by subclassing `Action`:

```python
class MyCustomAction(Action):
    def execute(self, agent, **kwargs) -> bool:
        # Custom logic here
        return True
```

Add the action to your FSM in the configuration.

## API Reference

Rome includes an HTTP API for agent introspection:

- `GET /agent`: Returns current state, context, and FSM details.

## Development

### Run Tests

```bash
pytest               # All tests
pytest --cov=rome    # With coverage report
pytest test/test_*.py  # Specific test file
```

### Format & Lint Code

```bash
black .
mypy rome/
flake8 rome/
```

## Contributing

Contributions will be accepted once the project reaches a stable state.

## License

See LICENSE file.