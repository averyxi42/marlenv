"""Self-play and training for the AlphaZero snake solver.

The value target is the discounted communal return, and the policy target is
the search's root visit counts marginalised per snake. Only the *communal*
value is supervised: the per-snake heads are combined by the objective and
the loss is taken on the result, so each head has to discover its own share.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from marlenv.policies.features import head_positions, observe
from marlenv.policies.objectives import get_objective


class ReplayBuffer:
    """Fixed-capacity store of self-play positions, bucketed by board size.

    A variable board size means views differ in height and width, which
    cannot be stacked into one batch. Positions are therefore grouped by view
    shape and each batch is drawn from a single bucket, so batches stay
    homogeneous without padding the boards -- padding would distort both the
    global pooling and the wall channel, since "outside the board" and "wall"
    are not the same thing to the network.

    Views are kept as uint8 because every feature plane is an indicator; they
    become float on the way into a batch.
    """

    def __init__(self, capacity=30000, seed=None):
        self.capacity = capacity
        self.buckets = {}
        self.total = 0
        self.random = random.Random(seed)

    def __len__(self):
        return self.total

    def add(self, views, alive, policy_target, value_target):
        entry = (views.astype(np.uint8), alive.astype(np.float32),
                 policy_target.astype(np.float32), np.float32(value_target))
        self.buckets.setdefault(views.shape, deque()).append(entry)
        self.total += 1
        while self.total > self.capacity:
            # evict from the largest bucket, so no board size is starved
            largest = max(self.buckets.values(), key=len)
            largest.popleft()
            self.total -= 1

    def sample(self, batch_size, device):
        # prefer buckets that can fill a whole batch, weighted by size
        keys = [k for k, v in self.buckets.items() if v]
        full = [k for k in keys if len(self.buckets[k]) >= batch_size]
        keys = full or keys
        weights = [len(self.buckets[k]) for k in keys]
        chosen = self.random.choices(keys, weights=weights)[0]
        bucket = list(self.buckets[chosen])

        batch = self.random.sample(bucket, min(batch_size, len(bucket)))
        views = np.stack([b[0] for b in batch]).astype(np.float32)
        alive = np.stack([b[1] for b in batch])
        policy = np.stack([b[2] for b in batch])
        value = np.array([b[3] for b in batch], dtype=np.float32)

        n_samples, n_snakes = views.shape[:2]
        flat = views.reshape(n_samples * n_snakes, *views.shape[2:])
        heads = head_positions(flat)
        return {
            'views': torch.from_numpy(flat).to(device),
            'heads': torch.from_numpy(heads).to(device),
            'alive': torch.from_numpy(alive).to(device),
            'policy': torch.from_numpy(policy).to(device),
            'value': torch.from_numpy(value).to(device),
            'shape': (n_samples, n_snakes),
        }


def self_play_episode(env, solver, objective, max_steps=120,
                      temperature_moves=8, rng=None):
    """Play one episode under the search, returning training positions.

    The first ``temperature_moves`` actions are sampled from the root visit
    distribution rather than taken greedily, so self-play games differ.
    """
    rng = rng or np.random.default_rng()
    objective = get_objective(objective)
    base = env.unwrapped
    solver.reset()

    views, alives, policies, rewards = [], [], [], []
    steps = 0
    bootstrap = 0.0
    for step in range(max_steps):
        all_idx = list(range(base.num_snakes))
        planes, _ = observe(env, all_idx)
        alive = np.array([s.alive for s in base.snakes], dtype=np.float32)

        action, target = solver.search(env, add_noise=True)
        if step < temperature_moves:
            action = _sample_action(solver, target, base, rng)

        _, rews, terminated, truncated, _ = env.step(action)
        views.append(planes)
        alives.append(alive)
        policies.append(target)
        rewards.append(float(objective.fold(rews)))
        steps += 1
        if all(terminated) or all(truncated):
            break
    else:
        # cut short by the step budget, so bootstrap the unseen tail
        bootstrap = float(solver.last_stats.get('root_value', 0.0))

    discount = solver.discount
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = bootstrap
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + discount * running
        returns[t] = running

    positions = list(zip(views, alives, policies, returns))
    return positions, {'steps': steps, 'communal_return': float(sum(rewards))}


def _sample_action(solver, target, base, rng):
    """Sample each snake's action from its marginal visit distribution."""
    action = [0] * base.num_snakes
    for i, snake in enumerate(base.snakes):
        probs = target[i]
        if snake.alive and probs.sum() > 0:
            action[i] = int(rng.choice(len(probs), p=probs / probs.sum()))
    return action


def compute_losses(net, batch, objective):
    """Communal value loss and masked per-snake policy loss."""
    objective = get_objective(objective)
    n_samples, n_snakes = batch['shape']
    logits, values = net(batch['views'], batch['heads'])
    logits = logits.view(n_samples, n_snakes, -1)
    values = values.view(n_samples, n_snakes)

    alive = batch['alive']
    # dead snakes are padded to zero and excluded from the fold
    values = values * alive
    communal = objective.torch_fold(values, alive)
    value_loss = F.mse_loss(communal, batch['value'])

    log_probs = F.log_softmax(logits, dim=-1)
    per_snake = -(batch['policy'] * log_probs).sum(dim=-1)
    denom = alive.sum().clamp(min=1.0)
    policy_loss = (per_snake * alive).sum() / denom
    return value_loss, policy_loss


def train_step(net, optimizer, batch, objective, policy_weight=1.0,
               grad_clip=5.0):
    net.train()
    value_loss, policy_loss = compute_losses(net, batch, objective)
    loss = value_loss + policy_weight * policy_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optimizer.step()
    net.eval()
    return {'loss': loss.item(), 'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()}
