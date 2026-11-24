"""Loads the implementation under test for Week 1.

Modules are loaded explicitly by path and registered in sys.modules, so
`from pe import ProcessingElement` inside the test file resolves to the
directory you chose with --impl, regardless of sys.path ordering.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["pe", "array_1d"]      # order matters: array_1d imports pe


def _load(name, directory):
    path = os.path.join(directory, f"{name}.py")
    if not os.path.isfile(path):
        raise SystemExit(f"could not find {name}.py in {directory}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pytest_configure(config):
    impl = config.getoption("--impl") or HERE
    impl = os.path.abspath(impl)
    if not os.path.isdir(impl):
        raise SystemExit(f"--impl directory does not exist: {impl}")
    for name in MODULES:
        _load(name, impl)
