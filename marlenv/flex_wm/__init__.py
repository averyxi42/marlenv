"""World action modelling over sets of observation/action pairs.

A generalisation of :mod:`marlenv.wm`: same weights and same rule, but the
agent count is free to move and attention scope varies by block.
"""
from marlenv.flex_wm.attention import AGENT, FRAME, GLOBAL, parse_schedule
from marlenv.flex_wm.model import FlexWorldModel
from marlenv.flex_wm.pairs import PairBatch

__all__ = ['AGENT', 'FRAME', 'GLOBAL', 'FlexWorldModel', 'PairBatch',
           'parse_schedule']
