import pytest
from hypothesis import given, strategies as st

# filepath: /d:/DeepLearning/手撕代码/test_字节面试算法题.py
from 字节面试算法题 import func

@pytest.mark.parametrize("n, A, expected", [
    # Basic test cases
    (123, [1, 2, 3, 4, 5], 123),
    (123, [1, 2], 122),
    (555, [1, 2, 3, 4, 5], 555),
    (789, [1, 2, 3, 4, 5], 555),
    
    # Edge cases
    (100, [1], 111),
    (999, [9], 999),
    (123, [0, 1, 2], 122),
    (1000, [1, 2, 3], 999),
    
    # Cases with expected -1
    (123, [], -1),
    (456, [6, 7, 8, 9], -1),
    
    # Cases with different array sizes
    (123, [1], 111),
    (456, [1, 2, 3, 4, 5, 6, 7, 8, 9], 456),
    
    # Cases with repeating digits
    (111, [1, 2], 111),
    (222, [1, 2], 222),
    
    # Cases with decreasing numbers
    (321, [1, 2, 3], 321),
    (654, [1, 2, 3, 4, 5, 6], 654),
    
    # Cases requiring backtracking
    (234, [1, 2, 3], 233),
    (345, [1, 3, 4], 344),
    
    # Large number cases
    (9876, [1, 2, 3, 4, 5, 6, 7, 8, 9], 9876),
    (1234, [1, 2], 1222)
])
def test_func_parametrize(n, A, expected):
    assert func(n, A) == expected

@given(
    n=st.integers(min_value=1, max_value=9999),
    A=st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=10)
)
def test_func_property(n, A):
    result = func(n, A)
    if result != -1:
        assert isinstance(result, int)
        assert len(str(result)) == len(str(n))
        
def test_empty_array():
    assert func(123, []) == -1

def test_single_digit_array():
    assert func(123, [1]) == 111

def test_no_solution():
    assert func(123, [4, 5, 6]) == -1