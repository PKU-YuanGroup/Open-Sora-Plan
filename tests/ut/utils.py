"""
We can't use assert in our code for codecheck, so create this auxiliary function to wrap
the assert case in ut for ci.
"""


def judge_expression(expression):
    if not expression:
        raise AssertionError


class TestConfig(object):
    def __init__(self, entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = TestConfig(v)
            else:
                self.__dict__[k] = v

    def to_dict(self):
        ret = {}
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret
