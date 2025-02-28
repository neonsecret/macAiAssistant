import os
from datetime import datetime
import inspect
import json
import subprocess
from typing import Optional, Dict, Any, get_type_hints

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
    def run_terminal_command(command_to_run: str) -> str:
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
            os.system(bash_script)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Launch error: {e.stderr}")
            return False


def create_tool_schema(cls: type) -> list:
    """Generate OpenAI-compatible tool schema from class static methods"""
    tools = []
    type_dict = {
        "int": "integer",
        "str": "string",
        "bool": "boolean"
    }

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        sig = inspect.signature(method)
        doc = inspect.getdoc(method) or f"Executes {name} function"

        params = {"type": "object", "properties": {}, "required": []}
        return_type = get_type_hints(method).get('return', 'str').__name__
        if return_type in type_dict.keys():
            return_type = type_dict[return_type]

        # Build parameter schema
        for param in sig.parameters.values():
            if param.name != 'self':
                param_type = str(param.annotation.__name__) if param.annotation != param.empty else 'string'
                if param_type in type_dict.keys():
                    param_type = type_dict[param_type]
                params['properties'][param.name] = {"type": param_type}
                if param.default == param.empty:
                    params['required'].append(param.name)

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": params,
                "returns": {"type": str(return_type)}
            }
        })

    return tools


def execute_function_call(tool_call: Dict[str, Any]) -> str:
    """Execute a function call using the AssistantFunctions class"""
    print(f"Tool call: {tool_call}")
    func_name = tool_call['function']['name']
    args = tool_call['function'].get('arguments', {})
    args = eval(args)
    print(f"Executing {func_name} with args {args}")

    if hasattr(AssistantFunctions, func_name):
        method = getattr(AssistantFunctions, func_name)
        return method(**args)
    raise ValueError(f"Function {func_name} not found in AssistantFunctions")


if __name__ == "__main__":
    tools = create_tool_schema(AssistantFunctions)
    for tool in tools:
        print(tool)
