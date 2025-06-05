import multiprocessing as mp
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from .agent import Agent
from .config import set_attributes_from_config
from .logger import get_logger


def _agent_worker(agent_name: str, agent_role: str, repository: str, config: Dict,
                  max_iterations: int, result_queue: mp.Queue):
    """Simplified worker with direct exception propagation"""
    agent = None

    try:
        result_queue.put(('starting', agent_name, 'Initializing'))

        agent = Agent(name=agent_name, role=agent_role, repository=repository, config=config)
        result_queue.put(('initialized', agent_name, 'Running'))

        results = agent.run_loop(max_iterations=max_iterations, raise_exception=True)
        results['agent_name'] = agent_name

        result_queue.put(('success', agent_name, results))

    except Exception as e:
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'agent_name': agent_name
        }
        result_queue.put(('error', agent_name, error_info))

    finally:
        if agent:
            try:
                agent.shutdown()
            except Exception as e:
                result_queue.put(('warning', agent_name, f'Shutdown error: {str(e)}'))


class MultiAgent:
    """Multi-agent manager with robust process handling"""

    def __init__(self, agent_role_json: str, repository: str, config: Dict = None):
        self.logger = get_logger()

        # Apply config
        multi_agent_config = config.get('MultiAgent', {}) if config else {}
        set_attributes_from_config(self, multi_agent_config, ['agent_role_json', 'repository'])

        # Override with parameters
        self.agent_role_json = agent_role_json or self.agent_role_json
        self.repository = repository or self.repository

        # Validate and load
        self._validate_inputs()
        self.agents_config = self._load_agents_config()
        self.base_config = config or {}

        # Runtime state
        self.agent_processes: Dict[str, mp.Process] = {}
        self.result_queue = mp.Queue()
        self.results: Dict = {}
        self.agent_status: Dict = {}

    def _validate_inputs(self):
        """Validate inputs, raise if invalid"""
        for path, name in [(self.agent_role_json, "Agent role JSON"), (self.repository, "Repository")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")

    def _load_agents_config(self) -> Dict[str, str]:
        """Load and validate agent configurations"""
        try:
            with open(self.agent_role_json, 'r') as f:
                config = json.load(f)

            if not isinstance(config, dict) or not config:
                raise ValueError("Agent config must be non-empty dictionary")

            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.agent_role_json}: {e}")

    def run_loop(self, max_iterations: int = 10, timeout_per_agent: int = 0) -> Dict:
        """Run all agents in parallel with enhanced error handling

        Args:
            max_iterations: Maximum iterations per agent
            timeout_per_agent: Timeout in seconds (0 = infinite timeout)
        """
        # Reset state
        for container in [self.results, self.agent_status, self.agent_processes]:
            container.clear()

        # Initialize status
        for name in self.agents_config:
            self.agent_status[name] = 'pending'

        # Start processes
        self._start_processes(max_iterations)

        # Collect results
        self._collect_results(timeout_per_agent)

        # Cleanup
        self._cleanup_processes()
        summary = self._create_summary()

        return {
            'total_agents': len(self.agents_config),
            'completed_agents': len(self.results),
            'agent_results': self.results,
            'agent_status': self.agent_status,
            'repository': self.repository,
            'log_dir': self.get_log_dir(),
            'summary': summary
        }

    def _start_processes(self, max_iterations: int):
        """Start agent processes with error handling"""
        for agent_name, agent_role in self.agents_config.items():
            try:
                process = mp.Process(
                    target=_agent_worker,
                    args=(agent_name, agent_role, self.repository, self.base_config,
                          max_iterations, self.result_queue)
                )
                process.start()
                self.agent_processes[agent_name] = process
                self.agent_status[agent_name] = 'starting'

            except Exception as e:
                self._mark_agent_failed(agent_name, 'ProcessStartError', f'Failed to start: {str(e)}')

    def _collect_results(self, timeout_per_agent: int):
        """Enhanced result collection with failure handling"""
        completed = 0
        total = len(self.agents_config)
        start_time = time.time()

        while completed < total:
            try:
                elapsed = time.time() - start_time
                remaining = max(30, timeout_per_agent - elapsed) if timeout_per_agent > 0 else None

                result_type, agent_name, data = self.result_queue.get(timeout=remaining)

                if result_type in ['starting', 'initialized']:
                    self.agent_status[agent_name] = 'initializing' if result_type == 'starting' else 'running'

                elif result_type == 'success':
                    self.results[agent_name] = data
                    self.agent_status[agent_name] = 'completed'
                    completed += 1

                elif result_type == 'error':
                    self.results[agent_name] = {'error': data}
                    self.agent_status[agent_name] = 'failed'

                    # Print full error details to console
                    error_type = data.get('error_type', 'Unknown')
                    error_msg = data.get('error_message', 'No message')
                    traceback_str = data.get('traceback', 'No traceback available')

                    self.logger.error(f"Agent '{agent_name}' failed: {error_type} - {error_msg}")
                    self.logger.error(f"Traceback:\n{traceback_str}")

                    completed += 1

                elif result_type == 'warning':
                    # Warnings are still logged but not printed to console
                    pass

            except Exception as e:
                if timeout_per_agent > 0:
                    for name, status in self.agent_status.items():
                        if status not in ['completed', 'failed', 'failed_to_start']:
                            self._mark_agent_failed(name, 'TimeoutError', f'Timed out after {timeout_per_agent}s')
                            completed += 1
                break

    def _mark_agent_failed(self, agent_name: str, error_type: str, error_message: str):
        """Mark agent as failed with error info"""
        self.results[agent_name] = {
            'error': {
                'error_type': error_type,
                'error_message': error_message,
                'agent_name': agent_name
            }
        }
        self.agent_status[agent_name] = 'failed_to_start' if 'start' in error_type.lower() else 'timeout'

        # Print error to console with traceback
        self.logger.error(f"Agent '{agent_name}' {error_type}: {error_message}")
        self.logger.error(f"Traceback:\n{traceback.format_exc()}")

    def _cleanup_processes(self):
        """Cleanup all processes with escalating termination"""
        for agent_name, process in self.agent_processes.items():
            if not process.is_alive():
                continue

            # Try graceful shutdown first
            process.join(timeout=10)

            # Escalate if needed
            for action, timeout in [('terminate', 5), ('kill', 0)]:
                if process.is_alive():
                    getattr(process, action)()
                    if timeout > 0:
                        process.join(timeout=timeout)

    def _create_summary(self) -> Dict:
        """Create execution summary"""
        if not self.results:
            return {"status": "No results available"}

        successful = sum(1 for r in self.results.values() if 'error' not in r)
        failed = len(self.results) - successful

        # Status breakdown
        status_counts = {}
        for status in self.agent_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1

        # Agent summaries
        agent_summaries = {}
        for name, result in self.results.items():
            if 'error' in result:
                error = result['error']
                agent_summaries[name] = {
                    "status": "FAILED",
                    "error_type": error.get('error_type', 'Unknown'),
                    "error_message": error.get('error_message', 'No message'),
                    "final_status": self.agent_status.get(name, 'unknown')
                }
            else:
                actions = result.get('execution_stats', {}).get('actions_executed', 0)
                agent_summaries[name] = {
                    "status": "SUCCESS",
                    "actions_executed": actions,
                    "final_state": result.get('agent_info', {}).get('current_state', 'Unknown'),
                    "final_status": self.agent_status.get(name, 'unknown')
                }

        summary = {
            "type": "MultiAgent Summary",
            "repository": self.repository,
            "total_agents": len(self.agents_config),
            "completed_agents": len(self.results),
            "successful_agents": successful,
            "failed_agents": failed,
            "success_rate": f"{(successful/len(self.results)*100):.1f}%" if self.results else "0.0%",
            "status_breakdown": status_counts,
            "agent_summaries": agent_summaries
        }

        # Save summary to file
        try:
            summary_file = Path(self.get_log_dir()) / "multi_agent_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")

        return summary

    def get_log_dir(self) -> str:
        """Get/create log directory"""
        from .config import LOG_DIR_NAME
        log_dir = Path(self.repository) / LOG_DIR_NAME
        log_dir.mkdir(exist_ok=True)
        return str(log_dir)

    def shutdown(self):
        """Graceful shutdown"""
        self._cleanup_processes()
        self.agent_processes.clear()
