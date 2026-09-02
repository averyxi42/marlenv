from marlenv.policies.mcts import MCTSSolver
from marlenv.policies.objectives import (CommunalObjective, OBJECTIVES,
                                         get_objective)

__all__ = ['MCTSSolver', 'CommunalObjective', 'OBJECTIVES', 'get_objective']

try:  # the learned solver is optional -- it needs torch
    from marlenv.policies.alphazero import AlphaZeroSolver
    from marlenv.policies.networks import NetworkEvaluator, SnakeNet
except ImportError:  # pragma: no cover - exercised only without torch
    pass
else:
    __all__ += ['AlphaZeroSolver', 'NetworkEvaluator', 'SnakeNet']
