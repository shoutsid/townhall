"""
Unit tests for the ReplayMemory class.
"""

from townhall.models.replay_memory import ReplayMemory, Transition
import pytest


@pytest.fixture
def replay_memory():
    return ReplayMemory(capacity=100)


def test_push(replay_memory):
    transition = Transition(
        state=[1, 2, 3], action=0, next_state=[4, 5, 6], reward=1)
    replay_memory.push(*transition)
    assert len(replay_memory) == 1


def test_sample(replay_memory):
    transition1 = Transition(
        state=[1, 2, 3], action=0, next_state=[4, 5, 6], reward=1)
    transition2 = Transition(
        state=[4, 5, 6], action=1, next_state=[7, 8, 9], reward=0)
    replay_memory.push(*transition1)
    replay_memory.push(*transition2)
    batch = replay_memory.sample(batch_size=1)
    assert len(batch) == 1
    assert isinstance(batch[0], Transition)


def test_capacity(replay_memory):
    for i in range(200):
        transition = Transition(
            state=[i], action=0, next_state=[i+1], reward=1)
        replay_memory.push(*transition)
    assert len(replay_memory) == 100
