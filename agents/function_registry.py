"""
This module contains a class to manage a registry of functions.

Attributes:
-----------
None

Classes:
--------
FunctionRegistry
  A class to manage a registry of functions.

Methods:
--------
None
"""


class FunctionRegistry:
    """
    A class to manage a registry of functions.

    Attributes:
    -----------
    functions : dict
      A dictionary to store the registered functions.

    Methods:
    --------
    add_function(name, func)
      Adds a function to the registry.

    execute_function(name, *args, **kwargs)
      Executes a function from the registry.

    list_functions()
      Returns a list of all the registered function names.
    """

    def __init__(self):
        """
        Initializes an empty FunctionRegistry object.
        """
        self.functions = {}

    def add_function(self, name, func):
        """
        Adds a function to the registry.

        Parameters:
        -----------
        name : str
          The name of the function to be added.
        func : function
          The function to be added to the registry.
        """
        self.functions[name] = func

    def execute_function(self, name, *args, **kwargs):
        """
        Executes a function from the registry.

        Parameters:
        -----------
        name : str
          The name of the function to be executed.
        *args : tuple
          Positional arguments to be passed to the function.
        **kwargs : dict
          Keyword arguments to be passed to the function.

        Returns:
        --------
        The return value of the executed function.
        """
        return self.functions[name](*args, **kwargs)

    def list_functions(self):
        """
        Returns a list of all the registered function names.

        Returns:
        --------
        A list of strings representing the names of all the registered functions.
        """
        return list(self.functions.keys())
