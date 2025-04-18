import inspect
import subprocess
import difflib
import os
from datetime import datetime
from typing import Dict, Any, get_type_hints
from .web_search import search_topic_RAG


class AssistantFunctions:
    @staticmethod
    def speak(message_to_user: str) -> str:
        """Respond directly to the user with the final answer or message."""
        return message_to_user

    ### Generic functions
    @staticmethod
    def get_current_time() -> str:
        """Get the current time in the current timezone"""
        return datetime.now().strftime("%H:%M")

    # @staticmethod
    # def get_current_date() -> str:
    #     """Get the current date (day, month, year)"""
    #     return datetime.now().strftime("%A the %d, %B %Y")

    @staticmethod
    def search_online(query_to_search: str) -> str:
        """Search a query online"""
        # results = DDGS().text(query_to_search, max_results=3)
        # return str(results)
        return search_topic_RAG(query_to_search)

    ### System-related functions
    @staticmethod
    def run_terminal_command(command_to_run: str) -> str:
        """Run an arbitrary terminal command"""
        # this says arbitrary for the AI helper to think so
        # if command_to_run.split()[0] in ["uname"]:
        if input(f"Allow {command_to_run} to be run? y/n") == "y":
            return (subprocess
                    .run(command_to_run.split(), stdout=subprocess.PIPE)
                    .stdout
                    .decode()
                    .replace("\n", ""))
        else:
            return "Forbidden by the user"

    @staticmethod
    def add_calendar_event(summary: str,
                           start_datetime: str,
                           end_datetime: str,
                           datetime_format: str = '%Y-%m-%d %H:%M') -> bool | str:
        """Add event to macOS Calendar using AppleScript

        This function uses subprocess to execute an AppleScript command that creates
        a new event in the default macOS Calendar. It handles date parsing and formatting
        to ensure compatibility with AppleScript's expected date format.

        The function first parses the provided start_time and end_time strings according
        to the specified datetime_format. If the format doesn't include year, month, or day,
        the function uses today's date for the missing components. It then converts these
        datetime objects to the specific format required by AppleScript.

        The function escapes special characters in the summary to prevent AppleScript
        execution errors, constructs the full AppleScript command, and executes it using
        subprocess.run().

        Args:
            summary: Title/summary of the calendar event
            start_datetime: Event start time as string
                        Missing date components default to today's date
            end_datetime: Event end time as string
                      Missing date components default to today's date
            datetime_format: Format string for parsing provided start_datetime and end_datetime,
             default isoformat is '%Y-%m-%d %H:%M'

        Returns:
            bool: True if event was successfully added, False otherwise

        Example:
            # Full datetime specification
            add_calendar_event("Team Meeting", "2025-04-01 14:30", "2025-04-01 15:30")

            # Time-only specification (uses today's date)
            add_calendar_event("Quick Call", "14:30", "15:00", "%H:%M")
        """

        # Get today's date components
        today = datetime.now()
        today_year = today.year
        today_month = today.month
        today_day = today.day

        # Check if format includes year, month, and day
        has_year = '%Y' in datetime_format or '%y' in datetime_format
        has_month = '%b' in datetime_format or '%B' in datetime_format or '%m' in datetime_format
        has_day = '%d' in datetime_format

        try:
            # Parse datetime strings
            parsed_start = datetime.strptime(start_datetime, datetime_format)
            parsed_end = datetime.strptime(end_datetime, datetime_format)

            # Replace missing components with today's date
            if not has_year:
                parsed_start = parsed_start.replace(year=today_year)
                parsed_end = parsed_end.replace(year=today_year)
            if not has_month:
                parsed_start = parsed_start.replace(month=today_month)
                parsed_end = parsed_end.replace(month=today_month)
            if not has_day:
                parsed_start = parsed_start.replace(day=today_day)
                parsed_end = parsed_end.replace(day=today_day)

            # Format dates for AppleScript
            # AppleScript date format: "month/day/year hour:minute:00 AM/PM"
            start_date_str = parsed_start.strftime("%-d/%-m/%Y %-I:%M:00 %p")
            end_date_str = parsed_end.strftime("%-d/%-m/%Y %-I:%M:00 %p")

            # Escape double quotes in the summary
            summary = summary.replace('"', '\\"')

            # Create AppleScript command
            applescript = f'''
            tell application "Calendar"
                tell calendar "Calendar"
                    make new event at end with properties {{summary:"{summary}", start date:date "{start_date_str}", end date:date "{end_date_str}"}}
                end tell
                save
            end tell
            '''

            # Execute AppleScript
            subprocess.run(['osascript', '-e', applescript],
                           capture_output=True,
                           text=True,
                           check=True)
            return True
        except ValueError as e:
            print(f"Error parsing date strings: {e}")
            return f"Error parsing date strings: {e}"
        except subprocess.CalledProcessError as e:
            print(f"Error adding event to calendar: {e.stderr}")
            return f"Error adding event to calendar: {e.stderr}"

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
            return True
        try:
            subprocess.run(['/opt/homebrew/bin/brew', 'install', package], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Installation error: {e.stderr}")
            return f"Installation error: {e.stderr}"

    @staticmethod
    def set_screen_brightness(level: float) -> bool | str:
        """
        Sets the screen brightness on macOS.

        Parameters:
        level (float): Brightness level from 0.0 (darkest) to 1.0 (brightest).
        """
        if not (0.0 <= level <= 1.0):
            raise ValueError("Brightness level must be between 0.0 and 1.0")

        try:
            subprocess.run(["brightness", str(level)], check=True)
            print(f"Brightness set to {level}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to set brightness: {e}")
            return f"Failed to set brightness: {e}"

    @staticmethod
    def find_app_name(app_query: str) -> str | None:
        """
        Search for an application matching the given name.
        Returns the correct app name for use with 'open -a', or None if not found.
        """
        search_dirs = [
            "/Applications",
            "/System/Applications",
            os.path.expanduser("~/Applications")
        ]
        app_names = set()

        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
            for item in os.listdir(directory):
                if item.endswith(".app"):
                    app_names.add(item[:-4])  # Remove '.app'

        # Attempt exact match (case-insensitive)
        for name in app_names:
            if name.lower() == app_query.lower():
                return name

        # Attempt close match
        matches = difflib.get_close_matches(app_query, app_names, n=1, cutoff=0.6)
        if matches:
            return matches[0]

        return "App not installed."

    @staticmethod
    def open_app(app_name: str) -> bool | str:
        """Launch application or website"""
        try:
            subprocess.run(['open', '-a', app_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Probably {app_name} is not installed")
            found_apps = AssistantFunctions.find_app_name(app_name)
            if found_apps != "App not installed.":
                try:
                    subprocess.run(['open', '-a', found_apps], check=True)
                    return True
                except:
                    pass
            return "App not installed."

    @staticmethod
    def open_url_default_browser(url: str) -> bool | str:
        """Launch application or website"""
        bash_script = f'osascript -e \'do shell script "open \\"{url}\\""\''
        try:
            os.system(bash_script)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Launch error: {e.stderr}")
            return f"Launch error: {e.stderr}"


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
        return_type = get_type_hints(method).get('return', 'str')
        try:
            return_type = return_type.__name__
        except:
            return_type = str(return_type)
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
    try:
        args = eval(args)
    except:
        return f"Failed to find the function {args}"
    print(f"Executing {func_name} with args {args}")

    if hasattr(AssistantFunctions, func_name):
        method = getattr(AssistantFunctions, func_name)
        return method(**args)
    raise ValueError(f"Function {func_name} not found in AssistantFunctions")


if __name__ == "__main__":
    tools = create_tool_schema(AssistantFunctions)
    for tool in tools:
        print(tool)
