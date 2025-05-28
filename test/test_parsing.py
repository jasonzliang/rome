#!/usr/bin/env python3
"""
Comprehensive tests for parsing utilities.
Run with: python test_parsing.py
"""

import ast
import json
import pytest
from typing import List, Dict, Any
import sys
import os

# Add the current directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all functions from the parsing module
from rome.parsing import (
    parse_code_cached, clear_ast_cache, get_cache_stats,
    parse_python_response, extract_function_names, extract_function_from_code,
    extract_comments_from_code, parse_json_response, extract_all_definitions
)


class TestFixtures:
    """Centralized test data for maximum reuse."""

    # Sample Python code snippets
    SIMPLE_FUNCTION = '''def hello_world():
    """A simple function."""
    print("Hello, World!")
    return "Hello"'''

    CLASS_WITH_METHODS = '''class Calculator:
    """A simple calculator class."""

    def __init__(self, value=0):
        self.value = value

    def add(self, x):
        """Add x to current value."""
        self.value += x
        return self.value

    def multiply(self, x):
        # Multiply current value by x
        self.value *= x
        return self.value'''

    COMPLEX_CODE = '''import math
from typing import List, Optional

def fibonacci(n: int) -> int:
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    """Process data efficiently."""

    def __init__(self, data: List[int]):
        self.data = data

    async def process_async(self) -> Optional[float]:
        """Process data asynchronously."""
        if not self.data:
            return None
        return sum(self.data) / len(self.data)

# Global variable
CONSTANT = 42'''

    # Markdown responses with code
    MARKDOWN_PYTHON = '''Here's some Python code:

```python
def test_function():
    return "test"
```

That should work!'''

    MARKDOWN_NO_LANG = '''Here's the code:

```
x = 5
print(x)
```'''

    INLINE_CODE = 'Use `import os` to work with files or `def func():` to define functions.'

    # JSON test data
    JSON_OBJECT = '{"name": "test", "value": 42}'
    JSON_ARRAY = '[1, 2, 3, {"nested": true}]'
    JSON_IN_MARKDOWN = '''Here's the JSON:

```json
{"result": "success", "data": [1, 2, 3]}
```'''

    JSON_IN_TEXT = 'The response is {"status": "ok", "count": 5} from the API.'


@pytest.fixture
def sample_codes():
    """Provide sample code snippets."""
    return TestFixtures()


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear AST cache before each test."""
    clear_ast_cache()
    yield
    clear_ast_cache()


class TestASTCaching:
    """Test AST parsing and caching functionality."""

    def test_parse_code_cached_basic(self, sample_codes):
        """Test basic AST parsing with caching."""
        tree1 = parse_code_cached(sample_codes.SIMPLE_FUNCTION)
        tree2 = parse_code_cached(sample_codes.SIMPLE_FUNCTION)

        assert isinstance(tree1, ast.AST)
        assert tree1 is tree2  # Should be same object from cache

        stats = get_cache_stats()
        assert stats['cache_size'] == 1
        assert stats['num_entries'] == 1

    def test_parse_code_cached_different_content(self, sample_codes):
        """Test caching with different code content."""
        tree1 = parse_code_cached(sample_codes.SIMPLE_FUNCTION)
        tree2 = parse_code_cached(sample_codes.CLASS_WITH_METHODS)

        assert tree1 is not tree2
        assert get_cache_stats()['cache_size'] == 2

    def test_parse_code_invalid_syntax(self):
        """Test parsing invalid Python syntax."""
        with pytest.raises(SyntaxError):
            parse_code_cached("def invalid syntax here")

    def test_clear_cache(self, sample_codes):
        """Test cache clearing."""
        parse_code_cached(sample_codes.SIMPLE_FUNCTION)
        assert get_cache_stats()['cache_size'] == 1

        clear_ast_cache()
        assert get_cache_stats()['cache_size'] == 0


class TestPythonResponseParsing:
    """Test extraction of Python code from various text formats."""

    @pytest.mark.parametrize("input_text,expected", [
        (TestFixtures.MARKDOWN_PYTHON, 'def test_function():\n    return "test"'),
        (TestFixtures.MARKDOWN_NO_LANG, 'x = 5\nprint(x)'),
        (TestFixtures.INLINE_CODE, 'import os'),  # First inline code match
        ('', None),
        ('No code here', None),
        ('```\n\n```', None),  # Empty code block
    ])
    def test_parse_python_response_cases(self, input_text, expected):
        """Test various Python response parsing scenarios."""
        result = parse_python_response(input_text)
        assert result == expected

    def test_parse_python_response_multiple_blocks(self):
        """Test parsing with multiple code blocks."""
        text = '''First block:
```python
def first():
    pass
```

Second block:
```
def second():
    pass
```'''
        result = parse_python_response(text)
        assert result == 'def first():\n    pass'  # Should get first Python block


class TestFunctionExtraction:
    """Test function name and code extraction."""

    def test_extract_function_names_basic(self, sample_codes):
        """Test basic function name extraction."""
        names = extract_function_names(sample_codes.SIMPLE_FUNCTION)
        assert names == ['hello_world']

    def test_extract_function_names_with_methods(self, sample_codes):
        """Test function extraction including class methods."""
        names = extract_function_names(sample_codes.CLASS_WITH_METHODS, include_methods=True)
        expected = ['Calculator.__init__', 'Calculator.add', 'Calculator.multiply']
        assert names == expected

    def test_extract_function_names_complex(self, sample_codes):
        """Test function extraction from complex code."""
        names = extract_function_names(sample_codes.COMPLEX_CODE, include_methods=True)
        expected = ['fibonacci', 'DataProcessor.__init__', 'DataProcessor.process_async']
        assert names == expected

    @pytest.mark.parametrize("code,expected", [
        ('', []),
        ('   ', []),
        ('def invalid syntax', []),  # Invalid syntax
        ('x = 5', []),  # No functions
    ])
    def test_extract_function_names_edge_cases(self, code, expected):
        """Test edge cases for function name extraction."""
        result = extract_function_names(code)
        assert result == expected

    def test_extract_function_from_code(self, sample_codes):
        """Test extracting specific function source code."""
        result = extract_function_from_code(sample_codes.SIMPLE_FUNCTION, 'hello_world')
        assert 'def hello_world():' in result
        assert 'print("Hello, World!")' in result

    def test_extract_method_from_code(self, sample_codes):
        """Test extracting class method source code."""
        result = extract_function_from_code(
            sample_codes.CLASS_WITH_METHODS,
            'Calculator.add',
            include_methods=True
        )
        assert result is not None
        assert 'def add(self, x):' in result
        assert 'self.value += x' in result

    @pytest.mark.parametrize("func_name,expected_none", [
        ('nonexistent', True),
        ('', True),
        ('Calculator.nonexistent', True),
    ])
    def test_extract_function_not_found(self, sample_codes, func_name, expected_none):
        """Test function extraction when function doesn't exist."""
        result = extract_function_from_code(sample_codes.CLASS_WITH_METHODS, func_name, include_methods=True)
        assert (result is None) == expected_none


class TestCommentExtraction:
    """Test comment and docstring extraction."""

    def test_extract_comments_docstrings_only(self, sample_codes):
        """Test extracting only docstrings."""
        result = extract_comments_from_code(sample_codes.CLASS_WITH_METHODS)
        assert 'A simple calculator class.' in result
        assert 'Add x to current value.' in result
        assert 'Multiply current value by x' not in result  # Single-line comment

    def test_extract_comments_including_single(self, sample_codes):
        """Test extracting all comments including single-line."""
        result = extract_comments_from_code(sample_codes.CLASS_WITH_METHODS, include_single_comments=True)
        assert 'A simple calculator class.' in result
        assert 'Add x to current value.' in result
        assert 'Multiply current value by x' in result

    def test_extract_comments_mixed_quotes(self):
        """Test extracting comments with different quote styles."""
        code = '''def func():
    """Double quote docstring."""
    pass

def func2():
    \'\'\'Single quote docstring.\'\'\'
    pass'''
        result = extract_comments_from_code(code)
        assert 'Double quote docstring.' in result
        assert 'Single quote docstring.' in result


class TestJSONParsing:
    """Test JSON response parsing with various formats."""

    @pytest.mark.parametrize("json_str,expected_type", [
        (TestFixtures.JSON_OBJECT, dict),
        (TestFixtures.JSON_ARRAY, list),
        ('{"nested": {"deep": [1, 2, 3]}}', dict),
        ('[{"mixed": true}, "array"]', list),
    ])
    def test_parse_json_direct(self, json_str, expected_type):
        """Test direct JSON parsing."""
        result = parse_json_response(json_str)
        assert isinstance(result, expected_type)

    def test_parse_json_from_markdown(self):
        """Test JSON extraction from markdown code blocks."""
        result = parse_json_response(TestFixtures.JSON_IN_MARKDOWN)
        expected = {"result": "success", "data": [1, 2, 3]}
        assert result == expected

    def test_parse_json_from_text(self):
        """Test JSON extraction from plain text."""
        result = parse_json_response(TestFixtures.JSON_IN_TEXT)
        expected = {"status": "ok", "count": 5}
        assert result == expected

    @pytest.mark.parametrize("invalid_input", [
        '',
        '   ',
        'No JSON here',
        '{"invalid": json}',  # Invalid syntax
        '{"unclosed": true',   # Unclosed brace
    ])
    def test_parse_json_invalid(self, invalid_input):
        """Test JSON parsing with invalid inputs."""
        result = parse_json_response(invalid_input)
        assert result is None


class TestDefinitionExtraction:
    """Test extraction of all function and class definitions."""

    def test_extract_all_definitions_comprehensive(self, sample_codes):
        """Test comprehensive definition extraction."""
        definitions = extract_all_definitions(sample_codes.COMPLEX_CODE)

        # Should find function and class
        names = [d['name'] for d in definitions]
        assert 'fibonacci' in names
        assert 'DataProcessor' in names

        # Check function details
        fib_def = next(d for d in definitions if d['name'] == 'fibonacci')
        assert fib_def['type'] == 'function'
        assert 'fibonacci number' in fib_def['docstring']
        assert fib_def['signature'] == 'def fibonacci(n: int) -> int'

        # Check class details
        class_def = next(d for d in definitions if d['name'] == 'DataProcessor')
        assert class_def['type'] == 'class'
        assert 'Process data efficiently' in class_def['docstring']
        assert '__init__' in class_def['methods']
        assert 'process_async' in class_def['methods']

    def test_extract_all_definitions_sorting(self, sample_codes):
        """Test that definitions are sorted by line number."""
        definitions = extract_all_definitions(sample_codes.COMPLEX_CODE)
        line_numbers = [d['line_number'] for d in definitions]
        assert line_numbers == sorted(line_numbers)

    @pytest.mark.parametrize("invalid_code", [
        '',
        '   ',
        'def invalid syntax here',
        'x = 5',  # No definitions
    ])
    def test_extract_all_definitions_edge_cases(self, invalid_code):
        """Test definition extraction edge cases."""
        result = extract_all_definitions(invalid_code)
        assert result == []


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self, sample_codes):
        """Test a complete workflow using multiple functions."""
        # 1. Parse markdown to get Python code
        markdown = f'''Here's a Python class:

```python
{sample_codes.CLASS_WITH_METHODS}
```'''

        python_code = parse_python_response(markdown)
        assert python_code is not None

        # 2. Extract function names
        functions = extract_function_names(python_code, include_methods=True)
        assert 'Calculator.add' in functions

        # 3. Extract specific function
        add_method = extract_function_from_code(python_code, 'Calculator.add', include_methods=True)
        assert 'def add(self, x):' in add_method

        # 4. Extract all definitions
        definitions = extract_all_definitions(python_code)
        class_def = next(d for d in definitions if d['name'] == 'Calculator')
        assert class_def['type'] == 'class'
        assert 'add' in class_def['methods']

    def test_cache_efficiency(self, sample_codes):
        """Test that caching improves performance for repeated parsing."""
        code = sample_codes.COMPLEX_CODE

        # First parsing populates cache
        clear_ast_cache()
        extract_function_names(code)
        assert get_cache_stats()['cache_size'] == 1

        # Subsequent operations should use cached AST
        extract_all_definitions(code)
        extract_function_from_code(code, 'fibonacci')
        assert get_cache_stats()['cache_size'] == 1  # Still only one cached entry


def run_tests():
    """Run all tests when script is executed directly."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
