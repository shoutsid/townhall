import sys
import os


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from agents.helpers.function_registry import FunctionRegistry
import pytest


class TestFunctionRegistry:
    def setup_method(self):
        self.registry = FunctionRegistry()

        def test_func():
            return "test"

        self.test_func = test_func

    def test_add_function(self):
        self.registry.add_function("test_func", self.test_func)
        assert "test_func" in self.registry.functions

    def test_execute_function(self):
        self.registry.add_function("test_func", self.test_func)
        result = self.registry.execute_function("test_func")
        assert result == "test"

    def test_list_functions(self):
        self.registry.add_function("test_func", self.test_func)
        functions = self.registry.list_functions()
        assert "test_func" in functions
