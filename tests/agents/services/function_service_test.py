import pytest
from agents.services.function_service import FunctionService


@pytest.fixture
def function_service():
    return FunctionService()


def test_add_function(function_service):
    function_body = "def addition(a, b): return a + b"
    response = function_service.add_function("addition", function_body)
    assert response == "Function 'addition' added!"
    result = function_service.execute_function("addition", 2, 3)
    assert result == 5


def test_execute_function(function_service):
    function_body = "def multiplication(a, b): return a * b"
    function_service.add_function("multiplication", function_body)
    result = function_service.execute_function("multiplication", 2, 3)
    assert result == 6


def test_add_multiple_functions(function_service):
    functions = {
        "subtract": "def subtract(a, b): return a - b",
        "divide": "def divide(a, b): return a / b",
    }
    response = function_service.add_multiple_functions(functions)
    assert response == "Function 'subtract' added!\nFunction 'divide' added!"
    result1 = function_service.execute_function("subtract", 5, 3)
    assert result1 == 2
    result2 = function_service.execute_function("divide", 10, 2)
    assert result2 == 5


def test_execute_multiple_functions(function_service):
    functions = {
        "subtract": "def subtract(a, b): return a - b",
        "divide": "def divide(a, b): return a / b",
    }
    function_service.add_multiple_functions(functions)
    function_calls = [
        {"name": "subtract", "args": [5, 3]},
        {"name": "divide", "kwargs": {"a": 10, "b": 2}},
    ]
    response = function_service.execute_multiple_functions(function_calls)
    assert response == "2\n5.0"
