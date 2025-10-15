"""Base Agent class providing core functionality for all agent types"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, Callable
import yaml

from .openai import OpenAIHandler
from .config import (DEFAULT_CONFIG, LOG_DIR_NAME, AGENT_NAME_LENGTH, LONGEST_SUMMARY_LEN,
                     set_attributes_from_config, merge_with_default_config, format_yaml_like)
from .logger import get_logger
from .parsing import parse_python_response, parse_json_response
from .process import process_managed
from .agent_memory import AgentMemory

yaml.add_representer(list, lambda dumper, data: dumper.represent_sequence(
    'tag:yaml.org,2002:seq', data, flow_style=True))


@process_managed
class BaseAgent:
    """Base agent providing core infrastructure for all agent types"""

    def __init__(self, name: str = None, role: str = None,
                 repository: str = None, config: Dict = None):
        """Initialize base agent with core components"""
        self.logger = get_logger()
        self._setup_config(config)
        self._validate_name_role(name, role)
        self._setup_repository_and_logging(repository)
        self._setup_openai_handler()
        self._setup_agent_memory()
        self._register_cleanup()
        self.export_config()

        self.logger.info(f"BaseAgent '{self.name}' initialized")

    def _setup_config(self, config_dict: Dict = None) -> None:
        """Setup and validate configuration"""
        self.config = merge_with_default_config(config_dict) if config_dict else DEFAULT_CONFIG.copy()
        agent_config = self.config.get('Agent', {})
        set_attributes_from_config(self, agent_config,
            ['name', 'role', 'repository', 'log_pid'])

    def _validate_name_role(self, name: str, role: str) -> None:
        """Validates and formats agent name and role"""
        if role: self.role = role
        if name: self.name = name
        self.logger.assert_true(self.role and self.name,
            f"Invalid name or role: {name}, {role}")

        if "your role" not in self.role.lower():
            self.role = f"Your role:\n{self.role}"

        a, b = AGENT_NAME_LENGTH
        clean_name = ''.join(c for c in self.name if c.isalnum())
        clean_name = clean_name.ljust(a, '0') if len(clean_name) < a else clean_name[:b]

        self.logger.assert_true(clean_name and a <= len(clean_name) <= b,
            f"Agent name must be {a}-{b} alphanumeric characters")
        self.name = clean_name

    def _setup_repository_and_logging(self, repository: str = None) -> None:
        """Validate repository and configure logging"""
        if repository: self.repository = str(repository)
        self.logger.assert_true(
            self.repository and os.path.exists(self.repository),
            f"Repository path does not exist: {self.repository}")

        log_config = self.config.get('Logger', {}).copy()
        if not log_config.get('base_dir'):
            log_config['base_dir'] = self.get_log_dir()
        if not log_config.get('filename'):
            log_config['filename'] = f"{self.get_id()}.console.log"
        get_logger().configure(log_config)

        self.logger.info(f"Logging configured: {log_config['base_dir']}")

    def _setup_openai_handler(self) -> None:
        """Initialize OpenAI handler"""
        openai_config = self.config.get('OpenAIHandler', {})
        self.openai_handler = OpenAIHandler(config=openai_config)
        self.logger.debug("OpenAI handler initialized")

    def _setup_agent_memory(self) -> None:
        """Initialize agent memory system"""
        memory_config = self.config.get('AgentMemory', {})

        self.agent_memory = AgentMemory(
            agent_name=self.get_id(pid=True),
            log_dir=self.get_log_dir(),
            config=memory_config
        )

        if self.agent_memory.is_enabled():
            self.logger.info(f"Agent memory enabled (auto_inject={self.agent_memory.auto_inject}, auto_remember={self.agent_memory.auto_remember})")
        else:
            self.logger.debug("Agent memory disabled")


    def _register_cleanup(self) -> None:
        """Register cleanup handlers"""
        self.shutdown_called = False

    def get_id(self, pid=False) -> str:
        """Unique identifier for agent in file system"""
        return f'agent_{self.name}_{os.getpid()}' if (self.log_pid or pid) else f'agent_{self.name}'

    def get_repo(self) -> str:
        """Get agent repository, creating if needed"""
        os.makedirs(self.repository, exist_ok=True)
        return self.repository

    def get_log_dir(self) -> str:
        """Get agent log directory, creating if needed"""
        log_dir = os.path.join(self.repository, LOG_DIR_NAME)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def export_config(self, filepath: str = None) -> None:
        """Export agent configuration to YAML file"""
        filepath = filepath or os.path.join(self.get_log_dir(), f"{self.get_id()}.config.yaml")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False, indent=4)
        self.logger.info(f"Configuration exported: {filepath}")

    def parse_json_response(self, response: str) -> Dict:
        """Parse JSON response"""
        result = parse_json_response(response)
        if result is None:
            self.logger.error("Failed to parse JSON from response")
        return result

    def parse_python_response(self, response: str) -> str:
        """Parse Python code from response"""
        result = parse_python_response(response)
        if result is None:
            self.logger.error("Failed to parse Python code from response")
        return result

    def shutdown(self) -> None:
        """Clean up resources before termination"""
        if self.shutdown_called: return
        self.shutdown_called = True
        try:
            self.logger.info("BaseAgent shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def remember(self, content: str, context: str = None, metadata: Dict = None) -> bool:
        """Store content in long-term memory"""
        return self.agent_memory.remember(content, context, metadata)

    def recall(self, query: str, context: str = None) -> str:
        """Query long-term memory and get most relevant results"""
        return self.agent_memory.recall(query, context)

    # def chat_completion(self, prompt: str, system_message: str = None,
    #                    override_config: Dict = None, response_format: Dict = None,
    #                    extra_body: Dict = None) -> str:
    #     """Direct access to chat completion with configuration options"""
    #     system_message = system_message or self.role
    #     return self.openai_handler.chat_completion(
    #         prompt=prompt, system_message=system_message,
    #         override_config=override_config, response_format=response_format,
    #         extra_body=extra_body)

    def chat_completion(self, prompt: str, system_message: str = None,
                       override_config: Dict = None, response_format: Dict = None,
                       extra_body: Dict = None) -> str:
        """Enhanced chat completion with automatic memory integration"""
        # Get memory context if enabled
        system_message = system_message or self.role

        memory_context = ""
        if self.agent_memory.is_enabled() and self.agent_memory.auto_inject:
            # Use prompt as query for recall
            memory_context = self.agent_memory.recall(prompt[:LONGEST_SUMMARY_LEN])

            if memory_context:
                # Inject memory into system message
                base_system = system_message or self.role
                enhanced_system = f"{base_system}\n\n[Relevant Memory Context]\n{memory_context}"
                system_message = enhanced_system
                self.logger.debug(f"Injected memory: {len(memory_context)} chars")

        # Call original chat completion
        response = self.openai_handler.chat_completion(
            prompt=prompt,
            system_message=system_message,
            override_config=override_config,
            response_format=response_format,
            extra_body=extra_body
        )

        # Auto-remember after getting response
        if self.agent_memory.is_enabled() and self.agent_memory.auto_remember:
            if self.agent_memory.should_remember(prompt, response):
                summary = \
                    f"Q: {prompt[:LONGEST_SUMMARY_LEN]}... A: {response[:LONGEST_SUMMARY_LEN]}..."
                self.agent_memory.remember(summary, context="interaction")

        return response
