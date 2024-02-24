import Queue
import pytest 

def test_head():
    queue = Queue([1, 2, 3, 4, 5])
    assert queue.head() == 5
