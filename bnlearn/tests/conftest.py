"""Pytest fixtures for bnlearn tests."""
import pytest


@pytest.fixture(autouse=True)
def _silence_bnlearn_logger():
    """Keep test output quiet (replaces the old verbose=0 default in calls)."""
    import bnlearn as bn
    previous = bn.get_logger().level
    bn.set_logger('silent')
    yield
    bn.get_logger().setLevel(previous)
