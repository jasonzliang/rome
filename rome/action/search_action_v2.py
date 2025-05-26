import glob
import os
import pprint
import random
import sys
import time
import traceback
from typing import Dict, Optional, Any, Union, List, Set, Tuple
from collections import defaultdict
import hashlib
from .action import Action
from ..logger import get_logger
from ..config import check_attrs
from ..versioning import save_original

class SearchActionV2(Action):
    """Enhanced action to comprehensively explore and search repository files"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        # Track exploration history across sessions
        self.exploration_history = set()
        self.file_interaction_scores = defaultdict(float)
        self.dependency_graph = defaultdict(set)

        # Enhanced config parameters
        check_attrs(self, [
            'max_files', 'file_types', 'exclude_dirs',
            'exclude_types', 'selection_criteria', 'batch_size',
            'exploration_strategy', 'diversity_weight', 'novelty_bonus',
            'dependency_analysis', 'semantic_clustering'
        ])

    def summary(self, agent) -> str:
        """Return a detailed summary of the search action"""
        explored_count = len(self.exploration_history)
        return (f"Comprehensive repository exploration: {', '.join(self.file_types)} files "
                f"(explored: {explored_count}, strategy: {self.exploration_strategy})")

    def _build_dependency_graph(self, files: List[str]) -> Dict[str, Set[str]]:
        """Build a comprehensive dependency graph between files"""
        self.logger.info("Building dependency graph for better exploration")
        dependencies = defaultdict(set)

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract imports and dependencies
                imports = self._extract_imports(content, file_path)
                dependencies[file_path].update(imports)

                # Extract function/class calls to other files
                cross_refs = self._extract_cross_references(content, files)
                dependencies[file_path].update(cross_refs)

            except Exception as e:
                self.logger.debug(f"Error analyzing dependencies for {file_path}: {e}")

        return dict(dependencies)

    def _extract_imports(self, content: str, file_path: str) -> Set[str]:
        """Extract import dependencies from file content"""
        imports = set()
        base_dir = os.path.dirname(file_path)

        for line in content.split('\n'):
            line = line.strip()

            # Handle relative imports
            if line.startswith('from .') or line.startswith('import .'):
                # Convert relative imports to file paths
                import_path = self._resolve_relative_import(line, base_dir)
                if import_path:
                    imports.add(import_path)

            # Handle absolute imports within the project
            elif 'import' in line and not line.startswith('#'):
                module_names = self._extract_module_names(line)
                for module in module_names:
                    potential_path = self._module_to_file_path(module, base_dir)
                    if potential_path:
                        imports.add(potential_path)

        return imports

    def _extract_cross_references(self, content: str, all_files: List[str]) -> Set[str]:
        """Find references to functions/classes defined in other files"""
        references = set()

        # Build a map of function/class names to files
        name_to_files = defaultdict(list)
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                for line in file_content.split('\n'):
                    line = line.strip()
                    if line.startswith('def ') or line.startswith('class '):
                        name = line.split('(')[0].split(':')[0].replace('def ', '').replace('class ', '')
                        name_to_files[name].append(file_path)
            except:
                continue

        # Find usage of these names in current content
        for line in content.split('\n'):
            for name, files in name_to_files.items():
                if name in line and not line.strip().startswith('#'):
                    references.update(files)

        return references

    def _calculate_exploration_diversity(self, files: List[str]) -> Dict[str, float]:
        """Calculate diversity scores to encourage exploration of different areas"""
        diversity_scores = {}

        # Group files by directory structure
        dir_groups = defaultdict(list)
        for file_path in files:
            dir_path = os.path.dirname(file_path)
            dir_groups[dir_path].append(file_path)

        # Calculate diversity based on directory exploration
        for file_path in files:
            dir_path = os.path.dirname(file_path)

            # Files in less explored directories get higher diversity scores
            explored_in_dir = sum(1 for f in dir_groups[dir_path] if f in self.exploration_history)
            total_in_dir = len(dir_groups[dir_path])

            # Diversity score: higher for unexplored directories
            if total_in_dir > 0:
                exploration_ratio = explored_in_dir / total_in_dir
                diversity_scores[file_path] = 1.0 - exploration_ratio
            else:
                diversity_scores[file_path] = 1.0

        return diversity_scores

    def _calculate_novelty_scores(self, files: List[str]) -> Dict[str, float]:
        """Calculate novelty scores based on content similarity and exploration history"""
        novelty_scores = {}

        for file_path in files:
            base_score = 1.0

            # Bonus for never explored files
            if file_path not in self.exploration_history:
                base_score += self.novelty_bonus

            # Penalty for recently explored files
            if file_path in self.exploration_history:
                # Reduce score based on how often it's been selected
                interaction_count = self.file_interaction_scores.get(file_path, 0)
                base_score = max(0.1, base_score - (interaction_count * 0.2))

            novelty_scores[file_path] = base_score

        return novelty_scores

    def _semantic_file_clustering(self, files: List[str], file_overviews: List[Dict]) -> Dict[str, str]:
        """Group files by semantic similarity to ensure diverse exploration"""
        clusters = {}
        cluster_names = ['core', 'utils', 'config', 'tests', 'data', 'models', 'actions', 'interfaces']

        for file_info in file_overviews:
            file_path = file_info['path']
            filename = os.path.basename(file_path).lower()
            dir_path = file_path.lower()

            # Assign to cluster based on naming patterns and content
            assigned_cluster = 'misc'

            if 'test' in filename or 'test' in dir_path:
                assigned_cluster = 'tests'
            elif 'config' in filename or 'setting' in filename:
                assigned_cluster = 'config'
            elif 'util' in filename or 'helper' in filename:
                assigned_cluster = 'utils'
            elif 'model' in filename or 'data' in dir_path:
                assigned_cluster = 'models'
            elif 'action' in filename or 'action' in dir_path:
                assigned_cluster = 'actions'
            elif any(core_term in filename for core_term in ['main', 'core', 'base', 'app']):
                assigned_cluster = 'core'
            elif 'interface' in filename or 'api' in filename:
                assigned_cluster = 'interfaces'

            clusters[file_path] = assigned_cluster

        return clusters

    def _multi_strategy_prioritization(self, agent, file_overviews: List[Dict]) -> List[Dict]:
        """Use multiple strategies to prioritize files for comprehensive exploration"""
        self.logger.info(f"Multi-strategy prioritization of {len(file_overviews)} files")

        files = [f['path'] for f in file_overviews]

        # Calculate various scoring components
        diversity_scores = self._calculate_exploration_diversity(files)
        novelty_scores = self._calculate_novelty_scores(files)
        file_clusters = self._semantic_file_clustering(files, file_overviews)

        # Build dependency graph if enabled
        if self.dependency_analysis:
            self.dependency_graph = self._build_dependency_graph(files)

        # Enhanced LLM prioritization with multiple factors
        prioritized_files = self._llm_enhanced_prioritization(
            agent, file_overviews, diversity_scores, novelty_scores, file_clusters
        )

        # Apply exploration strategy
        final_priorities = self._apply_exploration_strategy(prioritized_files, diversity_scores, novelty_scores)

        return final_priorities

    def _llm_enhanced_prioritization(self, agent, file_overviews: List[Dict],
                                   diversity_scores: Dict, novelty_scores: Dict,
                                   file_clusters: Dict) -> List[Dict]:
        """Enhanced LLM prioritization with additional context"""

        # Create cluster exploration summary
        cluster_exploration = defaultdict(int)
        for file_path in self.exploration_history:
            if file_path in file_clusters:
                cluster_exploration[file_clusters[file_path]] += 1

        prompt = f"""Prioritize {len(file_overviews)} files for comprehensive codebase exploration.

EXPLORATION CONTEXT:
- Total files explored so far: {len(self.exploration_history)}
- Selection criteria: {self.selection_criteria}
- Exploration strategy: {self.exploration_strategy}

CLUSTER EXPLORATION STATUS:
"""
        for cluster, count in cluster_exploration.items():
            total_in_cluster = sum(1 for c in file_clusters.values() if c == cluster)
            prompt += f"- {cluster}: {count}/{total_in_cluster} files explored\n"

        prompt += "\nFILES TO PRIORITIZE:\n"

        for i, file_info in enumerate(file_overviews):
            file_path = file_info['path']
            diversity = diversity_scores.get(file_path, 0)
            novelty = novelty_scores.get(file_path, 0)
            cluster = file_clusters.get(file_path, 'misc')
            explored = "✓" if file_path in self.exploration_history else "○"

            prompt += f"\n--- File {i+1}: {file_path} [{explored}] ---\n"
            prompt += f"Cluster: {cluster} | Diversity: {diversity:.2f} | Novelty: {novelty:.2f}\n"
            prompt += f"Size: {file_info['size_kb']} KB | Age: {file_info['modified_age']:.0f}s\n"

            if file_info['definitions']:
                prompt += "Key definitions: " + ", ".join(file_info['definitions'][:5]) + "\n"

        prompt += f"""
Return JSON array with ALL {len(file_overviews)} files prioritized for {self.exploration_strategy} exploration:

[
  {{
    "file_path": "path/to/file.py",
    "priority": 1-5,
    "exploration_value": 1-5,
    "reason": "Brief rationale considering diversity, novelty, and strategic value"
  }}
]

Prioritize based on:
1. Strategic importance for understanding the codebase
2. Diversity (unexplored areas/clusters)
3. Novelty (unvisited or rarely visited files)
4. Dependencies and architectural significance
"""

        response = agent.chat_completion(prompt=prompt, system_message=agent.role)
        result = agent.parse_json_response(response)

        if isinstance(result, list):
            return result
        else:
            # Fallback prioritization
            return self._fallback_prioritization(file_overviews, diversity_scores, novelty_scores)

    def _apply_exploration_strategy(self, prioritized_files: List[Dict],
                                  diversity_scores: Dict, novelty_scores: Dict) -> List[Dict]:
        """Apply specific exploration strategy to final prioritization"""

        if self.exploration_strategy == 'breadth_first':
            # Favor high diversity, spread across clusters
            for file_info in prioritized_files:
                file_path = file_info.get('file_path')
                if file_path:
                    diversity_bonus = diversity_scores.get(file_path, 0) * self.diversity_weight
                    file_info['priority'] = file_info.get('priority', 1) + diversity_bonus

        elif self.exploration_strategy == 'depth_first':
            # Follow dependencies and related files
            for file_info in prioritized_files:
                file_path = file_info.get('file_path')
                if file_path and file_path in self.dependency_graph:
                    dependency_bonus = len(self.dependency_graph[file_path]) * 0.5
                    file_info['priority'] = file_info.get('priority', 1) + dependency_bonus

        elif self.exploration_strategy == 'novelty_seeking':
            # Strongly favor unexplored files
            for file_info in prioritized_files:
                file_path = file_info.get('file_path')
                if file_path:
                    novelty_bonus = novelty_scores.get(file_path, 0) * 2.0
                    file_info['priority'] = file_info.get('priority', 1) + novelty_bonus

        elif self.exploration_strategy == 'adaptive':
            # Balance multiple factors based on exploration progress
            exploration_ratio = len(self.exploration_history) / max(len(prioritized_files), 1)

            for file_info in prioritized_files:
                file_path = file_info.get('file_path')
                if file_path:
                    base_priority = file_info.get('priority', 1)

                    if exploration_ratio < 0.3:  # Early exploration - favor diversity
                        bonus = diversity_scores.get(file_path, 0) * 1.5
                    elif exploration_ratio < 0.7:  # Mid exploration - balance factors
                        bonus = (diversity_scores.get(file_path, 0) + novelty_scores.get(file_path, 0))
                    else:  # Late exploration - focus on gaps and dependencies
                        bonus = novelty_scores.get(file_path, 0) * 2.0

                    file_info['priority'] = base_priority + bonus

        # Sort by final priority
        prioritized_files.sort(key=lambda x: x.get('priority', 0), reverse=True)
        return prioritized_files

    def _comprehensive_batch_processing(self, agent, file_paths: List[str],
                                      prioritized_files: List[Dict]) -> Dict:
        """Enhanced batch processing with better exploration coverage"""
        self.logger.info("Starting comprehensive batch processing")

        selected_files = []
        processed_clusters = set()

        current_batch = 0
        max_selections = min(3, len(file_paths) // 10 + 1)  # Select multiple files per session

        while (current_batch * self.batch_size < len(file_paths) and
               len(selected_files) < max_selections):

            start_idx = current_batch * self.batch_size
            end_idx = min((current_batch + 1) * self.batch_size, len(file_paths))
            batch_paths = file_paths[start_idx:end_idx]

            self.logger.info(f"Processing batch {current_batch+1}: files {start_idx+1}-{end_idx}")

            # Load batch data with enhanced metadata
            batch_data = self._load_enhanced_batch_data(batch_paths, prioritized_files)

            if not batch_data:
                current_batch += 1
                continue

            # Create enhanced selection prompt
            prompt = self._create_enhanced_selection_prompt(
                batch_data, current_batch, len(selected_files), max_selections
            )

            # Get LLM selection
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role,
                response_format={"type": "json_object"}
            )

            result = agent.parse_json_response(response)

            if result and 'selected_files' in result:
                batch_selections = result['selected_files']
                if not isinstance(batch_selections, list):
                    batch_selections = [batch_selections] if batch_selections else []

                for selection in batch_selections:
                    if isinstance(selection, dict) and 'file_number' in selection:
                        file_num = selection['file_number']
                        if 1 <= file_num <= len(batch_data):
                            file_data = batch_data[file_num - 1]
                            selected_files.append({
                                'path': file_data['path'],
                                'content': file_data['content'],
                                'reason': selection.get('reason', 'Selected for exploration'),
                                'priority': file_data['priority'],
                                'exploration_value': selection.get('exploration_value', 3)
                            })

            current_batch += 1

        # Return the highest value selection
        if selected_files:
            best_selection = max(selected_files, key=lambda x: x.get('exploration_value', 0))

            # Update exploration tracking
            self.exploration_history.add(best_selection['path'])
            self.file_interaction_scores[best_selection['path']] += 1

            return best_selection

        return None

    def _load_enhanced_batch_data(self, batch_paths: List[str], prioritized_files: List[Dict]) -> List[Dict]:
        """Load batch data with enhanced metadata for better selection"""
        batch_data = []

        for file_path in batch_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find priority info
                priority_info = next((f for f in prioritized_files if f.get("file_path") == file_path), {})

                # Calculate content hash for similarity detection
                content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

                batch_data.append({
                    'path': file_path,
                    'content': content,
                    'priority': priority_info.get('priority', 1),
                    'exploration_value': priority_info.get('exploration_value', 1),
                    'priority_reason': priority_info.get('reason', 'No reason provided'),
                    'content_hash': content_hash,
                    'previously_explored': file_path in self.exploration_history,
                    'interaction_count': self.file_interaction_scores.get(file_path, 0),
                    'line_count': len(content.split('\n')),
                    'function_count': content.count('def '),
                    'class_count': content.count('class ')
                })

            except Exception as e:
                self.logger.debug(f"Error loading {file_path}: {e}")

        return batch_data

    def _create_enhanced_selection_prompt(self, batch_data: List[Dict], current_batch: int,
                                        current_selections: int, max_selections: int) -> str:
        """Create enhanced selection prompt with exploration context"""

        prompt = f"""COMPREHENSIVE CODEBASE EXPLORATION - Batch {current_batch + 1}

EXPLORATION STATUS:
- Files explored so far: {len(self.exploration_history)}
- Selections this session: {current_selections}/{max_selections}
- Strategy: {self.exploration_strategy}
- Criteria: {self.selection_criteria}

BATCH FILES ({len(batch_data)} files):
"""

        for i, file_info in enumerate(batch_data):
            filename = os.path.basename(file_info['path'])
            explored_marker = "✓" if file_info['previously_explored'] else "○"

            prompt += f"""
File {i+1}: {filename} [{explored_marker}]
- Path: {file_info['path']}
- Priority: {file_info['priority']}/5 | Exploration Value: {file_info['exploration_value']}/5
- Size: {file_info['line_count']} lines | Functions: {file_info['function_count']} | Classes: {file_info['class_count']}
- Previous interactions: {file_info['interaction_count']}
- Reason: {file_info['priority_reason']}
"""

        prompt += "\n\nFULL FILE CONTENTS:\n"
        for i, file_info in enumerate(batch_data):
            prompt += f"\n{'='*60}\nFile {i+1}: {file_info['path']}\n{'='*60}\n"
            prompt += file_info['content']

        prompt += f"""

Select the most valuable file(s) for exploration. Consider:
1. Strategic importance for understanding the codebase
2. Unexplored areas (○ files preferred over ✓ files)
3. Architectural significance and dependencies
4. Code complexity and learning potential

Respond with JSON:
{{
    "selected_files": [
        {{
            "file_number": <int>,
            "path": "<file_path>",
            "reason": "<detailed rationale>",
            "exploration_value": <1-5>
        }}
    ],
    "exploration_notes": "<insights about this batch>"
}}

Select 1-2 files max per batch. Favor unexplored files unless explored files have exceptional strategic value.
"""

        return prompt

    def _fallback_prioritization(self, file_overviews: List[Dict], diversity_scores: Dict,
                                novelty_scores: Dict) -> List[Dict]:
        """Fallback prioritization when LLM fails"""
        prioritized_files = []

        for file_info in file_overviews:
            file_path = file_info['path']

            # Calculate composite score
            base_score = 1.0
            diversity_bonus = diversity_scores.get(file_path, 0) * self.diversity_weight
            novelty_bonus = novelty_scores.get(file_path, 0) * self.novelty_bonus

            # Size and age factors
            size_factor = max(0.1, 1.0 - (file_info.get('size_kb', 0) / 100))  # Prefer smaller files
            age_factor = min(2.0, file_info.get('modified_age', 0) / 86400)  # Prefer older files

            total_score = base_score + diversity_bonus + novelty_bonus + size_factor + age_factor

            prioritized_files.append({
                'file_path': file_path,
                'priority': min(5, max(1, int(total_score))),
                'exploration_value': min(5, max(1, int(total_score * 1.2))),
                'reason': f'Calculated priority (div:{diversity_bonus:.1f}, nov:{novelty_bonus:.1f})'
            })

        prioritized_files.sort(key=lambda x: x['priority'], reverse=True)
        return prioritized_files

    def execute(self, agent, **kwargs) -> bool:
        """Execute enhanced search with comprehensive exploration"""
        self.logger.info("Starting enhanced SearchAction execution")

        # Input validation
        if not hasattr(agent, 'openai_handler') or agent.openai_handler is None:
            raise ValueError("Agent must have an openai_handler attribute initialized")

        # File discovery phase
        all_files = self._discover_all_files(agent)
        if not all_files:
            self.logger.warning("No files found for exploration")
            return False

        # Apply filters
        filtered_files = self._apply_all_filters(all_files)
        self.logger.info(f"Found {len(filtered_files)} files after filtering")

        # Create comprehensive file overview
        file_overviews = self._create_global_overview(agent, filtered_files)

        # Multi-strategy prioritization
        prioritized_files = self._multi_strategy_prioritization(agent, file_overviews)

        # Comprehensive batch processing
        file_paths = [f.get('file_path') for f in prioritized_files if f.get('file_path')]
        selected_file = self._comprehensive_batch_processing(agent, file_paths, prioritized_files)

        if selected_file:
            self._update_exploration_tracking(selected_file)
            agent.context['selected_file'] = selected_file
            save_original(selected_file['path'], selected_file['content'])

            self.logger.info(f"Exploration completed. Selected: {selected_file['path']}")
            self.logger.info(f"Total exploration progress: {len(self.exploration_history)} files")
            return True
        else:
            agent.context['selected_file'] = None
            self.logger.warning("No suitable file selected for exploration")
            return False

    def _discover_all_files(self, agent) -> List[str]:
        """Discover all files in the repository"""
        all_files = []
        for file_type in self.file_types:
            if not file_type.startswith('.'):
                file_type = '.' + file_type
            search_path = os.path.join(agent.repository, f'**/*{file_type}')
            files = glob.glob(search_path, recursive=True)
            all_files.extend(files)
        return all_files

    def _apply_all_filters(self, files: List[str]) -> List[str]:
        """Apply all filtering steps"""
        filtered = self._filter_excluded_dirs(files)
        filtered = self._filter_excluded_types(filtered)
        filtered = self._filter_max_limit(filtered)
        return filtered

    def _update_exploration_tracking(self, selected_file: Dict):
        """Update exploration tracking data"""
        file_path = selected_file['path']
        self.exploration_history.add(file_path)
        self.file_interaction_scores[file_path] += 1

    # Keep existing filter methods unchanged
    def _filter_excluded_dirs(self, files: List[str]) -> List[str]:
        """Filter out files from excluded directories"""
        if not self.exclude_dirs:
            return files

        filtered_files = []
        for file_path in files:
            exclude = False
            for excluded_dir in self.exclude_dirs:
                normalized_path = file_path.replace('\\', '/')
                normalized_excluded = excluded_dir.replace('\\', '/')
                if f'/{normalized_excluded}/' in normalized_path or \
                   normalized_path.startswith(f'{normalized_excluded}/'):
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(file_path)
        return filtered_files

    def _filter_excluded_types(self, files: List[str]) -> List[str]:
        """Filter out files with excluded file types"""
        if not self.exclude_types:
            return files

        filtered_files = []
        for file_path in files:
            exclude = any(file_path.endswith(exclude_type) for exclude_type in self.exclude_types)
            if not exclude:
                filtered_files.append(file_path)
        return filtered_files

    def _filter_max_limit(self, files: List[str]) -> List[str]:
        """Apply maximum file limit with random sampling"""
        if len(files) > self.max_files:
            random.shuffle(files)
            files = files[:self.max_files]
        return files

    def _create_global_overview(self, agent, files: List[str]) -> List[Dict]:
        """Create enhanced global overview with more metadata"""
        self.logger.info(f"Creating enhanced overview of {len(files)} files")
        file_overviews = []
        current_time = time.time()

        for file_path in files:
            try:
                stats = os.stat(file_path)
                size_kb = stats.st_size / 1024
                modified_age = current_time - stats.st_mtime

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                definitions = []
                imports = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('def ') or line.startswith('class '):
                        definitions.append(line)
                    elif 'import ' in line and not line.startswith('#'):
                        imports.append(line)

                file_overviews.append({
                    "path": file_path,
                    "size_kb": round(size_kb, 2),
                    "modified_age": round(modified_age, 1),
                    "definitions": definitions[:10],
                    "imports": imports[:5],
                    "line_count": len(content.split('\n')),
                    "complexity_estimate": len(definitions) + len(imports)
                })

            except Exception as e:
                self.logger.debug(f"Error reading {file_path}: {e}")

        return file_overviews

    # Utility methods for import resolution
    def _resolve_relative_import(self, import_line: str, base_dir: str) -> Optional[str]:
        """Resolve relative import to file path"""
        # Simplified implementation - can be enhanced
        return None

    def _extract_module_names(self, import_line: str) -> List[str]:
        """Extract module names from import statement"""
        modules = []
        if 'import ' in import_line:
            # Handle "from X import Y" and "import X" patterns
            if import_line.strip().startswith('from '):
                # from module import something
                parts = import_line.split('import')[0].replace('from ', '').strip()
                if parts and not parts.startswith('.'):
                    modules.append(parts)
            elif import_line.strip().startswith('import '):
                # import module1, module2
                import_part = import_line.split('import ')[1]
                for module in import_part.split(','):
                    module = module.strip().split(' as ')[0].split('.')[0]
                    if module and not module.startswith('.'):
                        modules.append(module)
        return modules

    def _module_to_file_path(self, module_name: str, base_dir: str) -> Optional[str]:
        """Convert module name to potential file path"""
        # Check if module corresponds to a local file
        potential_paths = [
            os.path.join(base_dir, f"{module_name}.py"),
            os.path.join(base_dir, module_name, "__init__.py"),
            os.path.join(os.path.dirname(base_dir), f"{module_name}.py"),
            os.path.join(os.path.dirname(base_dir), module_name, "__init__.py")
        ]

        for path in potential_paths:
            if os.path.exists(path):
                return path
        return None

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get comprehensive exploration statistics"""
        return {
            'total_explored': len(self.exploration_history),
            'exploration_history': list(self.exploration_history),
            'file_interaction_scores': dict(self.file_interaction_scores),
            'dependency_graph_size': len(self.dependency_graph),
            'most_visited_files': sorted(
                self.file_interaction_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def reset_exploration_history(self):
        """Reset exploration tracking - useful for fresh starts"""
        self.exploration_history.clear()
        self.file_interaction_scores.clear()
        self.dependency_graph.clear()
        self.logger.info("Exploration history reset")

    def suggest_next_exploration_targets(self, agent, top_n: int = 5) -> List[Dict]:
        """Suggest next files to explore based on current exploration state"""
        all_files = self._discover_all_files(agent)
        filtered_files = self._apply_all_filters(all_files)

        # Find unexplored files
        unexplored = [f for f in filtered_files if f not in self.exploration_history]

        if not unexplored:
            return []

        # Create quick overview
        file_overviews = self._create_global_overview(agent, unexplored[:50])  # Limit for performance

        # Calculate scores
        diversity_scores = self._calculate_exploration_diversity(unexplored)
        novelty_scores = self._calculate_novelty_scores(unexplored)

        suggestions = []
        for file_info in file_overviews:
            file_path = file_info['path']
            composite_score = (
                diversity_scores.get(file_path, 0) * 0.4 +
                novelty_scores.get(file_path, 0) * 0.3 +
                (file_info.get('complexity_estimate', 0) / 10) * 0.3
            )

            suggestions.append({
                'path': file_path,
                'score': composite_score,
                'diversity': diversity_scores.get(file_path, 0),
                'novelty': novelty_scores.get(file_path, 0),
                'complexity': file_info.get('complexity_estimate', 0),
                'size_kb': file_info.get('size_kb', 0),
                'reason': f"High exploration value (div:{diversity_scores.get(file_path, 0):.2f}, nov:{novelty_scores.get(file_path, 0):.2f})"
            })

        # Sort by composite score and return top N
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:top_n]
