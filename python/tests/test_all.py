import pytest
import metagene


def test_sum_as_string():
    assert metagene.sum_as_string(1, 1) == "2"
