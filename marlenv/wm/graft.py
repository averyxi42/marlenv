"""Growing a trained model deeper without throwing away what it learnt.

A residual block that contributes nothing is a block the network behaves as
though it did not have. So a deeper model can be started from a shallower
one exactly: copy each trained block, interleave a duplicate after it, and
silence the duplicate by zeroing what it writes back to the residual
stream. Step zero computes precisely what the shallow model computed, and
every duplicate then learns a correction from a starting point that is
already a useful transform rather than noise.

Silencing means the output projections only. The weights inside a duplicate
are the trained ones, so once its gate opens it has real features to work
with -- which is the whole reason to copy rather than initialise fresh.
Note the consequence, though: with the output zeroed, the gradient reaching
those inner weights is zero on the first step and afterwards scales with how
far the output has moved from zero. The duplicates wake up rather than start
running, which is the usual behaviour of a zero-initialised residual branch.
"""
import torch
import torch.nn as nn


def last_linear(module):
    """The layer a sublayer writes its result through."""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            return layer
    raise ValueError('no linear layer to silence')


def silence(block):
    """Zero what a block adds to the residual stream, and only that.

    Both halves have to go quiet, and a bias counts: zeroing a weight while
    leaving a bias behind leaves the block emitting a constant, which is not
    inert.
    """
    for sublayer in (block.attn, block.mlp):
        linear = last_linear(sublayer)
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


def source_depth(state):
    indices = [int(name.split('.')[1]) for name in state
               if name.startswith('blocks.')]
    if not indices:
        raise ValueError('checkpoint has no blocks')
    return max(indices) + 1


def graft_depth(model, state):
    """Load ``state`` into ``model``, interleaving copies if it is deeper.

    Equal depths are an ordinary load. Otherwise the target depth must be a
    whole multiple of the source, and each trained block is followed by its
    silenced duplicates.
    """
    shallow = source_depth(state)
    deep = len(model.blocks)
    if deep == shallow:
        model.load_state_dict(state)
        return 0
    if deep % shallow:
        raise ValueError(f'cannot graft depth {shallow} into {deep}: '
                         'the target depth must be a whole multiple')
    copies = deep // shallow

    grown = {}
    for name, value in state.items():
        if not name.startswith('blocks.'):
            grown[name] = value
            continue
        _, index, rest = name.split('.', 2)
        for copy in range(copies):
            grown[f'blocks.{int(index) * copies + copy}.{rest}'] = \
                value.clone()
    model.load_state_dict(grown)

    silenced = 0
    for index in range(shallow):
        for copy in range(1, copies):
            silence(model.blocks[index * copies + copy])
            silenced += 1
    return silenced


@torch.no_grad()
def preserves(shallow_model, deep_model, *args, **kwargs):
    """True when the grafted model still computes what the original did."""
    was = shallow_model.training, deep_model.training
    shallow_model.eval(), deep_model.eval()
    before = shallow_model(*args, **kwargs)
    after = deep_model(*args, **kwargs)
    shallow_model.train(was[0]), deep_model.train(was[1])
    if isinstance(before, tuple):
        return all(torch.allclose(a, b, atol=1e-6)
                   for a, b in zip(before, after))
    return torch.allclose(before, after, atol=1e-6)
