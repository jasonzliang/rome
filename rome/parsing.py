import json
import re
import ast
from typing import Dict, Optional, Any, Union, List


def parse_python_response(response: str) -> Optional[str]:
    """
    Extract the first Python code block from a response.

    Args:
        response (str): The response text containing code blocks

    Returns:
        Optional[str]: The first Python code block found, or None if no code found
    """
    if not response:
        return None

    # Pattern to match Python code blocks with various formats
    # Handles ```python, ``` python, and plain ``` code blocks
    python_patterns = [
        r'```\s*python\s*\n(.*?)\n```',  # ```python\n...\n```
        r'```\n(.*?)\n```'               # ```\n...\n```
    ]

    for pattern in python_patterns:
        # Find all matches with the current pattern
        matches = re.findall(pattern, response, re.DOTALL)

        # If we found any matches with this pattern
        if matches:
            # Clean up the code blocks by stripping leading/trailing whitespace
            python_blocks = [match.strip() for match in matches]

            # Return first non-empty block (minimal validation)
            for block in python_blocks:
                if block.strip():
                    return block

    # If no code blocks found, try to find inline code with single backticks
    # This is a fallback for simple code snippets
    inline_pattern = r'`([^`]+)`'
    inline_matches = re.findall(inline_pattern, response)

    # Only consider inline code with Python keywords
    python_keywords = ['def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while ', 'return ']
    for match in inline_matches:
        # Simple validation that this might be Python code
        if any(keyword in match for keyword in python_keywords) or '=' in match or 'print(' in match:
            return match.strip()

    # No valid Python code found
    return None


def parse_json_response(response: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Parse JSON response with optimized approach for both direct JSON and extraction.

    Handles both responses from JSON mode (response_format={"type": "json_object"})
    and extracts JSON from text responses when needed.

    Args:
        response (str): The response text, either direct JSON or containing JSON fragments

    Returns:
        Union[Dict[str, Any], List[Any]]: The parsed JSON object/array or None if parsing failed
    """
    if not response or not response.strip():
        return None

    # STEP 1: Try direct JSON parsing first (for response_format="json_object" responses)
    try:
        parsed_json = json.loads(response)
        # Return the parsed JSON as is, whether it's a dict or a list
        return parsed_json
    except json.JSONDecodeError:
        # Direct parsing failed, move to extraction methods
        pass

    # STEP 2: Extract JSON from code blocks
    json_block_patterns = [
        r'```\s*json\s*\n([\s\S]*?)\n```',  # ```json\n...\n```
        r'```\n([\s\S]*?)\n```'             # ```\n...\n```
    ]

    for pattern in json_block_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                cleaned_match = match.strip()
                parsed_json = json.loads(cleaned_match)
                # Return the parsed JSON as is, whether it's a dict or a list
                return parsed_json
            except json.JSONDecodeError:
                continue

    # STEP 3: Extract JSON from free text (balanced bracket approach)
    # Find objects with properly balanced braces
    start_indexes = []
    candidate_objects = []
    depth = 0

    for i, char in enumerate(response):
        if char == '{':
            if depth == 0:
                start_indexes.append(i)
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_indexes:
                candidate_objects.append(response[start_indexes.pop():i+1])
        # Also check for JSON arrays
        elif char == '[':
            if depth == 0:
                start_indexes.append(i)
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start_indexes:
                candidate_objects.append(response[start_indexes.pop():i+1])

    # Try to parse each candidate JSON object or array
    for candidate in candidate_objects:
        try:
            parsed_json = json.loads(candidate)
            # Return the parsed JSON regardless of type
            return parsed_json
        except json.JSONDecodeError:
            continue

    # No valid JSON found
    return None


def extract_all_definitions(content: str) -> List[Dict[str, str]]:
    """
    Extract both function and class definitions from Python code.

    Args:
        content (str): Python source code as a string

    Returns:
        List[Dict[str, str]]: List of dictionaries containing definition info.
                             Each dict has keys: 'name', 'signature', 'docstring', 'line_number', 'type'
                             Functions also have 'type': 'function'
                             Classes also have 'type': 'class' and 'methods': list
    """
    if not content:
        return []

    definitions = []

    try:
        # Parse the code into an AST
        tree = ast.parse(content)

        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function
                func_name = node.name
                func_signature = _get_function_signature(node)
                docstring = ast.get_docstring(node) or ""
                line_number = node.lineno

                definitions.append({
                    'name': func_name,
                    'signature': func_signature,
                    'docstring': docstring.strip(),
                    'line_number': line_number,
                    'type': 'function'
                })

            elif isinstance(node, ast.ClassDef):
                # Extract class
                class_name = node.name
                class_signature = _get_class_signature(node)
                docstring = ast.get_docstring(node) or ""
                line_number = node.lineno

                # Extract method names
                methods = []
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods.append(child.name)

                definitions.append({
                    'name': class_name,
                    'signature': class_signature,
                    'docstring': docstring.strip(),
                    'line_number': line_number,
                    'type': 'class',
                    'methods': methods
                })

    except SyntaxError:
        # If AST parsing fails, fall back to regex-based extraction
        definitions = _extract_all_definitions_regex(content)

    # Sort by line number
    definitions.sort(key=lambda x: x['line_number'])
    return definitions


def _get_function_signature(func_node: ast.FunctionDef) -> str:
    """
    Extract the complete function signature from an AST FunctionDef node.

    Args:
        func_node (ast.FunctionDef): The function definition node

    Returns:
        str: The function signature as a string
    """
    # Start with function name
    signature = f"def {func_node.name}("

    # Handle arguments
    args = []

    # Regular arguments
    for arg in func_node.args.args:
        arg_str = arg.arg
        # Add type annotation if present
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}" if hasattr(ast, 'unparse') else f": {_ast_to_string(arg.annotation)}"
        args.append(arg_str)

    # *args
    if func_node.args.vararg:
        vararg_str = f"*{func_node.args.vararg.arg}"
        if func_node.args.vararg.annotation:
            vararg_str += f": {ast.unparse(func_node.args.vararg.annotation)}" if hasattr(ast, 'unparse') else f": {_ast_to_string(func_node.args.vararg.annotation)}"
        args.append(vararg_str)

    # **kwargs
    if func_node.args.kwarg:
        kwarg_str = f"**{func_node.args.kwarg.arg}"
        if func_node.args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(func_node.args.kwarg.annotation)}" if hasattr(ast, 'unparse') else f": {_ast_to_string(func_node.args.kwarg.annotation)}"
        args.append(kwarg_str)

    signature += ", ".join(args)
    signature += ")"

    # Add return type annotation if present
    if func_node.returns:
        signature += f" -> {ast.unparse(func_node.returns)}" if hasattr(ast, 'unparse') else f" -> {_ast_to_string(func_node.returns)}"

    return signature


def _ast_to_string(node) -> str:
    """
    Convert an AST node to string representation (fallback for older Python versions).

    Args:
        node: AST node

    Returns:
        str: String representation of the node
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Attribute):
        return f"{_ast_to_string(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        return f"{_ast_to_string(node.value)}[{_ast_to_string(node.slice)}]"
    else:
        # For other types, return a generic representation
        return str(type(node).__name__)


def _get_class_signature(class_node: ast.ClassDef) -> str:
    """
    Extract the complete class signature from an AST ClassDef node.

    Args:
        class_node (ast.ClassDef): The class definition node

    Returns:
        str: The class signature as a string
    """
    # Start with class name
    signature = f"class {class_node.name}"

    # Handle inheritance
    if class_node.bases or class_node.keywords:
        args = []

        # Base classes
        for base in class_node.bases:
            if hasattr(ast, 'unparse'):
                args.append(ast.unparse(base))
            else:
                args.append(_ast_to_string(base))

        # Keyword arguments (like metaclass)
        for keyword in class_node.keywords:
            if hasattr(ast, 'unparse'):
                value = ast.unparse(keyword.value)
            else:
                value = _ast_to_string(keyword.value)
            args.append(f"{keyword.arg}={value}")

        if args:
            signature += f"({', '.join(args)})"

    signature += ":"
    return signature


def _extract_classes_regex(content: str) -> List[Dict[str, str]]:
    """
    Fallback regex-based class extraction when AST parsing fails.

    Args:
        content (str): Python source code as a string

    Returns:
        List[Dict[str, str]]: List of class information dictionaries
    """
    classes = []
    lines = content.split('\n')

    # Pattern to match class definitions
    class_pattern = r'^\s*class\s+(\w+)(?:\s*\([^)]*\))?\s*:'

    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(class_pattern, line)

        if match:
            class_name = match.group(1)
            signature = line.strip()
            line_number = i + 1

            # Look for docstring
            docstring = ""
            j = i + 1

            # Skip empty lines and find the first non-empty line
            while j < len(lines) and not lines[j].strip():
                j += 1

            # Check if the next non-empty line starts a docstring
            if j < len(lines):
                stripped_line = lines[j].strip()
                if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                    quote_type = '"""' if stripped_line.startswith('"""') else "'''"

                    # Single line docstring
                    if stripped_line.count(quote_type) >= 2:
                        docstring = stripped_line.replace(quote_type, '').strip()
                    else:
                        # Multi-line docstring
                        docstring_lines = [stripped_line.replace(quote_type, '')]
                        j += 1

                        while j < len(lines):
                            if quote_type in lines[j]:
                                docstring_lines.append(lines[j].split(quote_type)[0])
                                break
                            else:
                                docstring_lines.append(lines[j])
                            j += 1

                        docstring = '\n'.join(docstring_lines).strip()

            # Extract method names (simplified)
            methods = []
            j = i + 1
            indent_level = len(line) - len(line.lstrip())

            while j < len(lines):
                method_line = lines[j]
                if not method_line.strip():
                    j += 1
                    continue

                method_indent = len(method_line) - len(method_line.lstrip())

                # If we're back at the same or lower indentation, we've left the class
                if method_indent <= indent_level and method_line.strip():
                    break

                # Look for method definitions
                method_match = re.match(r'^\s*def\s+(\w+)', method_line)
                if method_match:
                    methods.append(method_match.group(1))

                j += 1

            classes.append({
                'name': class_name,
                'signature': signature,
                'docstring': docstring,
                'line_number': line_number,
                'type': 'class',
                'methods': methods
            })

        i += 1

    return classes


def _extract_all_definitions_regex(content: str) -> List[Dict[str, str]]:
    """
    Fallback regex-based extraction for both functions and classes.

    Args:
        content (str): Python source code as a string

    Returns:
        List[Dict[str, str]]: List of definition information dictionaries
    """
    definitions = []

    # Extract functions
    functions = _extract_functions_regex(content)
    for func in functions:
        func['type'] = 'function'
    definitions.extend(functions)

    # Extract classes
    classes = _extract_classes_regex(content)
    definitions.extend(classes)

    # Sort by line number
    definitions.sort(key=lambda x: x['line_number'])
    return definitions
