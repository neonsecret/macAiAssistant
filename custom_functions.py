from datetime import datetime
import inspect
import json


class AssistantFunctions:
    def get_time(self):
        return datetime.now().strftime("It's %H %M")

    def get_date(self):
        return datetime.now().strftime("It's %A the %d, %B %Y")


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


def convert_func_args():
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


if __name__ == "__main__":
    tools, choice = convert_func_args()
    print("Tools:")
    print(json.dumps(tools, indent=4))
    print("\nTool Choice:")
    print(json.dumps(choice, indent=4))
