import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.function_registry import FunctionRegistry


def test_add_function():
    registry = FunctionRegistry()

    def test_func():
        return "test"

    registry.add_function("test_func", test_func)
    assert "test_func" in registry.functions


def test_execute_function():
    registry = FunctionRegistry()

    def test_func():
        return "test"

    registry.add_function("test_func", test_func)
    result = registry.execute_function("test_func")
    assert result == "test"


def test_list_functions():
    registry = FunctionRegistry()

    def test_func():
        return "test"

    registry.add_function("test_func", test_func)
    functions = registry.list_functions()
    assert "test_func" in functions
