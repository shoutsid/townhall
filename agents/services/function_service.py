import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

from agents.helpers.function_registry import FunctionRegistry


class FunctionService:
    """
    A service for adding and executing functions dynamically.

    Attributes:
    registry (FunctionRegistry): A registry for storing added functions.
    """

    def __init__(self):
        self.registry = FunctionRegistry()

    def add_function(self, name, function_body):
        """
        Adds a new function to the registry.

        Args:
          name (str): The name of the function.
          function_body (str): The body of the function.

        Returns:
          str: A message indicating the function was added successfully.

        Raises:
          SyntaxError: If the function_body contains invalid syntax.

        Example:
          >>> function_service.add_function("my_function(x, y)", "return x + y")
          "Function 'my_function(x, y)' added!"
        """
        local_scope = {}
        try:
            compiled_func = compile(function_body, "<string>", "exec")
            exec(compiled_func, None, local_scope)
        except SyntaxError as e:
            raise SyntaxError(f"Invalid syntax in function body: {e}") from e
        new_func = local_scope[
            name.split("(")[0].strip()
        ]  # Assuming the name is the first word in the function_body
        self.registry.add_function(name, new_func)

        return f"Function '{name}' added!"

    def execute_function(self, name, *args, **kwargs):
        """
        Executes a function from the registry.

        Args:
        name (str): The name of the function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

        Returns:
        Any: The return value of the executed function.
        """
        return self.registry.execute_function(name, *args, **kwargs)

    def add_multiple_functions(self, functions):
        """
        Adds multiple functions to the registry.

        Args:
        functions (dict): A dictionary of function names and bodies.

        Returns:
        str: A message indicating all functions were added successfully.
        """
        responses = []
        for name, function_body in functions.items():
            response = self.add_function(name, function_body)
            responses.append(response)
        return "\n".join(responses)

    def execute_multiple_functions(self, function_calls):
        """
        Executes multiple functions from the registry.

        Args:
        function_calls (list): A list of dictionaries containing function names, positional arguments, and keyword arguments.

        Returns:
        str: A message indicating all functions were executed successfully.
        """
        responses = []
        for call in function_calls:
            name = call["name"]
            args = call.get("args", [])
            kwargs = call.get("kwargs", {})
            response = self.execute_function(name, *args, **kwargs)
            responses.append(str(response))
        return "\n".join(responses)
