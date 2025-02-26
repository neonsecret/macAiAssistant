import os
from datetime import datetime
import inspect
import json
import subprocess
from typing import Optional, Dict, Any

import xml.etree.ElementTree as ET
from xml.dom import minidom


class AssistantFunctions:
    ### Generic functions
    @staticmethod
    def get_time() -> str:
        return datetime.now().strftime("It's %H %M")

    @staticmethod
    def get_date() -> str:
        return datetime.now().strftime("It's %A the %d, %B %Y")

    @staticmethod
    def search_online(query_to_search: str) -> str:
        return "Searched."

    ### System-related functions
    @staticmethod
    def run_other_terminal_command(command_to_run: str) -> str:
        if command_to_run.split()[0] in ["uname"]:
            return os.system(command_to_run)

    @staticmethod
    def add_calendar_event(summary: str,
                           start_time: datetime,
                           end_time: datetime,
                           calendar_name: str = "Home") -> bool:
        """Add event to macOS Calendar using AppleScript"""
        try:
            apple_script = f'''
            tell application "Calendar"
                tell calendar "{calendar_name}"
                    make new event with properties {{
                        summary:"{summary}",
                        start date:(date "{start_time.strftime('%Y-%m-%d %H:%M:%S')}"),
                        end date:(date "{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    }}
                end tell
            end tell
            '''
            subprocess.run(['osascript', '-e', apple_script], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Calendar error: {e.stderr}")
            return False

    @staticmethod
    def set_volume(level: int) -> bool:
        """Set system volume (0-100)"""
        try:
            subprocess.run(['osascript', '-e', f'set volume output volume {level}'], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Volume error: {e.stderr}")
            return False

    @staticmethod  # TODO add search and interaction
    def install_via_brew(package: str) -> bool:
        """Install application via Homebrew or pkg file"""

        def _has_brew_tool(tool_name: str) -> bool:
            """Check if Homebrew tool is installed"""
            try:
                subprocess.run(['which', tool_name], check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError:
                return False

        if _has_brew_tool(package):
            return "Tool already installed"
        try:
            subprocess.run(['/opt/homebrew/bin/brew', 'install', package], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Installation error: {e.stderr}")
            return False

    @staticmethod
    def open_app(app_name: str) -> bool:
        """Launch application or website"""
        try:
            subprocess.run(['open', '-a', app_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Launch error: {e.stderr}")
            return False

    @staticmethod
    def open_url_default_browser(url: str) -> bool:
        """Launch application or website"""
        bash_script = f'osascript -e \'do shell script "open \\"{url}\\""\''
        try:
            subprocess.run(bash_script, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Launch error: {e.stderr}")
            return False


def convert_class_functions_to_dict(cls):
    tools = []
    for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip special and private methods
        if name.startswith("__") and name.endswith("__"):
            continue

        # Inspect the signature of the function to get parameters
        signature = inspect.signature(func)
        parameters = {
            "type": "object",
            "title": name,
            "properties": {},
            "required": []
        }

        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue  # Skip 'self' parameter

            param_type = "string"  # Default type

            # Map Python types to JSON types
            annotation = param.annotation
            if annotation != inspect.Parameter.empty:
                if annotation == int:
                    param_type = "integer"
                elif annotation == float:
                    param_type = "number"
                elif annotation == bool:
                    param_type = "boolean"
                elif annotation == list:
                    param_type = "array"
                elif annotation == dict:
                    param_type = "object"
                # Add more type mappings as needed

            parameters["properties"][param_name] = {
                "title": param_name.capitalize(),
                "type": param_type
            }

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        function_dict = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": parameters
            }
        }

        tools.append(function_dict)

    return tools


def convert_func_args_llama():
    tools = convert_class_functions_to_dict(AssistantFunctions)
    if tools:
        tool_choice = [{
            "type": "function",
            "function": {
                "name": tool["function"]["name"]  # Selecting the first function as an example
            }
        } for tool in tools]
    else:
        tool_choice = {}
    return tools, tool_choice
    # Display the tools and the selected tool choice


def class_to_r1_function_schema(cls: type) -> str:
    """
    Convert a Python class with static methods to an R1 function calling XML schema.

    This function iterates over the class's __dict__, extracts static methods,
    retrieves each method's signature, docstring, parameter types, defaults,
    and return type annotations and builds an XML representation.

    Args:
        cls: The class containing static methods.

    Returns:
        A pretty-printed XML string compatible with R1 function calling.
    """
    # Map Python type annotations to XML schema types.
    type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    root = ET.Element("tools")

    # Iterate through class attributes using __dict__.
    for name, member in cls.__dict__.items():
        # Check if the member is a static method.
        if not isinstance(member, staticmethod):
            continue

        # Get the underlying function.
        func = member.__func__

        # Extract the function's signature and docstring.
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        tool = ET.SubElement(root, "tool")
        ET.SubElement(tool, "name").text = name

        # Use the first line of the docstring as the description.
        description_text = doc.splitlines()[0] if doc else ""
        ET.SubElement(tool, "description").text = description_text

        parameters = ET.SubElement(tool, "parameters")

        # Process all parameters defined in the function.
        for param in sig.parameters.values():
            param_node = ET.SubElement(parameters, "parameter")
            ET.SubElement(param_node, "name").text = param.name

            # Map the type annotation if it exists; default to "string".
            param_type = type_mapping.get(param.annotation, "string")
            ET.SubElement(param_node, "type").text = param_type

            # Extract parameter description from the docstring.
            param_desc = parse_param_description(doc, param.name)
            if not param_desc:
                param_desc = f"{param.name} parameter"
            ET.SubElement(param_node, "description").text = param_desc

            if param.default != inspect.Parameter.empty:
                ET.SubElement(param_node, "default").text = str(param.default)

        # Handle the return annotation if provided.
        if sig.return_annotation != inspect.Signature.empty and sig.return_annotation is not None:
            returns = ET.SubElement(tool, "returns")
            return_type = type_mapping.get(sig.return_annotation, "object")
            ET.SubElement(returns, "type").text = return_type
            return_desc = parse_return_description(doc)
            if not return_desc:
                return_desc = "Function result"
            ET.SubElement(returns, "description").text = return_desc

    # Generate a pretty-printed XML string.
    xml_str = ET.tostring(root, encoding="unicode")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    return pretty_xml


def parse_param_description(doc: str, param_name: str) -> str:
    """
    Extract the description for a parameter from the docstring.

    Expects the docstring to contain lines formatted as:
      :param <param_name>: <description>

    Args:
        doc: The complete docstring.
        param_name: Name of the parameter.

    Returns:
        The extracted description for the parameter, or an empty string.
    """
    if not doc:
        return ""
    lines = doc.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith(f":param {param_name}:"):
            # Extract everything after the param definition.
            return line.split(":", 2)[-1].strip()
    return ""


def parse_return_description(doc: str) -> str:
    """
    Extract the return description from the docstring.

    Expects the docstring to contain a line formatted as:
      :return: <description>

    Args:
        doc: The complete docstring.

    Returns:
        The extracted return description, or an empty string.
    """
    if not doc:
        return ""
    lines = doc.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith(":return:"):
            return line.split(":", 1)[-1].strip()
    return ""


if __name__ == "__main__":
    print(class_to_r1_function_schema(AssistantFunctions))
