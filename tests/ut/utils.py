"""
We can't use assert in our code for codecheck, so create this auxiliary function to wrap
the assert case in ut for ci.
"""

def judge_expression(expression):
    if not expression:
        raise AssertionError
