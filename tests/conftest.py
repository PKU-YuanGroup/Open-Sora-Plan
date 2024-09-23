# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/tests/conftest.py
# reworked/refactored some parts to make it run.
import pytest


def pytest_configure(config):
    config.option.color = "yes"
    config.option.durations = 0
    config.option.durations_min = 1
    config.option.verbose = True


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


# We allow DistributedTest to reuse distributed environments. When the last
# test for a class is run, we want to make sure those distributed environments
# are destroyed.
def pytest_runtest_teardown(item, nextitem):
    if getattr(item.cls, "reuse_dist_env", False) and not nextitem:
        dist_test_class = item.cls()
        for num_procs, pool in dist_test_class._pool_cache.items():
            dist_test_class._close_pool(pool, num_procs, force=True)


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if getattr(fixturedef.func, "is_dist_fixture", False):
        dist_fixture_class = fixturedef.func()
        dist_fixture_class(request)


# we still want to configure path argument by ourselves
# for different prefix_name of different scripts so we use this method.
# One more thing, as you can see, it has more scalibility.
def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--baseline-json", action="store", default=None,
                    help="Path to the baseline JSON file")
    parser.addoption("--generate-log", action="store", default=None,
                    help="Path to the generate log file")
    parser.addoption("--generate-json", action="store", default=None,
                    help="Path to the generate JSON file")


@pytest.fixture(autouse=True)
def baseline_json(request: pytest.FixtureRequest):
    return request.config.getoption("--baseline-json")


@pytest.fixture(autouse=True)
def generate_log(request: pytest.FixtureRequest):
    return request.config.getoption("--generate-log")


@pytest.fixture(autouse=True)
def generate_json(request: pytest.FixtureRequest):
    return request.config.getoption("--generate-json")

