"""
    Transition model representing a transition in the simulation.
"""

from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))
