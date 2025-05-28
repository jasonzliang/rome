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

- Python 3.11+
- OpenAI API key

### Installation

#### From Source (Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rome.git
   cd rome
   ```

2. Install in development mode:
   ```bash
   # Basic installation
   pip install -e .

   # With development tools
   pip install -e .[dev]
   ```

#### From PyPI (Coming Soon)

```bash
# When published to PyPI
pip install rome
```

### Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

### Configuration

The library uses a YAML-based configuration system. Create a `config.yaml` file or use the default configuration:

```python
# Generate a default configuration
python -c "from config import generate_default_config; generate_default_config()"
```

Key configuration options:

- OpenAI model and parameters
- Repository path
- FSM type (minimal or simple)
- Logging settings
- Action-specific configurations

### Basic Usage

```python
from agent import Agent
from fsm import FSM_FACTORY

# Initialize agent with configuration
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

# Run the agent
results = agent.run_loop(max_iterations=10)
```

### Visualizing Agent State

Run the Streamlit app to visualize the agent's FSM and current state:

```bash
streamlit run web_app/streamlit_app.py
```

## Advanced Features

### Code Versioning

The library automatically versions code files as they are modified, maintaining a history of changes in a `.rome` directory next to each file.

### Custom FSM

You can create custom FSM configurations by implementing your own creation function and registering it:

```python
def create_custom_fsm(config):
    # Create a custom FSM configuration
    # ...

# Register your custom FSM
FSM_FACTORY['custom'] = create_custom_fsm
```

### Extending with New Actions

1. Create a new action class inheriting from `Action`
2. Implement the `execute` method
3. Add the action to your FSM configuration

```python
class MyCustomAction(Action):
    def execute(self, agent, **kwargs) -> bool:
        # Implementation
        return True
```

## API Reference

The library provides an HTTP API for monitoring and interacting with the agent:

- `GET /agent`: Get current agent state, context, and FSM information

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rome

# Run specific test file
pytest test/test_single_agent.py
```

### Code Formatting

```bash
# Format code with Black
black .

# Type checking with MyPy
mypy rome/

# Linting with flake8
flake8 rome/
```

## Contributing

Contributions will be allowed once code is in a stable state.

## License

See LICENSE file