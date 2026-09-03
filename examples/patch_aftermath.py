"""Add aftermath views to an already-collected dataset, in place.

An episode stores its seed and a full state per frame, so nothing has to be
re-simulated: resetting an env with the stored seed regenerates the very
same noise fields, and the stored grids put the board back. Reconstructed
views come out identical to the stored ones bit for bit, which this script
checks before changing anything.

What changes, for the frame after each agent's death:

* ``observations`` becomes the ordinary view from the cell the agent died
  entering, rather than the zeros the old env produced for the dead;
* ``poses`` becomes that viewpoint, rather than -1;
* ``cardinal_actions`` becomes the heading actually taken into that cell.
  It was an all-zero one-hot, which argmax reads as UP -- a confident claim
  that every dead snake kept driving north.
"""
import argparse
import os

import numpy as np
from datasets import load_from_disk

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.observation import egocentric_crop
from marlenv.core.snake import Cell, Direction, Snake
from marlenv.data import build_dataset, decode_episode
from marlenv.data.state import snake_bodies
from marlenv.grading.poses import LEFT_TURN, RIGHT_TURN

HEADINGS = list(Direction)
REWARD_DICT = {'fruit': 1.0, 'kill': 0.0, 'lose': -5.0, 'win': 0.0,
               'time': 0.01}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='marlenv/demodata')
    p.add_argument('--components', nargs='+', default=['expert', 'explore'])
    p.add_argument('--suffix', default='',
                   help='write to <name><suffix>; empty overwrites in place')
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--observation-noise', type=float, default=2.0)
    p.add_argument('--snake-noise', type=float, default=8.0)
    p.add_argument('--background-gradient', type=float, default=16.0)
    p.add_argument('--check', type=int, default=40,
                   help='episodes to verify reconstruction on first')
    return p.parse_args()


def make_env(row, args):
    return gym.make(
        'Snake-v1', height=int(row['height']), width=int(row['width']),
        num_snakes=int(row['num_agents']), num_fruits=args.num_fruits,
        reward_dict=REWARD_DICT, view_radius=int(row['view_radius']),
        observation_noise=args.observation_noise,
        snake_noise_sigma=args.snake_noise,
        background_gradient=args.background_gradient,
        gradient_period=6, noise_period=3,
        obstacle_density=args.obstacle_density, disable_env_checker=True)


def apply_state(env, content, body_index):
    """Put a stored state back into an env."""
    base = env.unwrapped
    grid = np.zeros_like(base.grid)
    grid[content == 1] = Cell.WALL.value
    grid[content == 2] = Cell.FRUIT.value

    bodies = snake_bodies(content, body_index)
    snakes = []
    for index in range(base.num_snakes):
        body = bodies.get(index)
        if not body or len(body) < 2:
            continue
        for coord in body:
            grid[coord] = Cell.BODY.value + 10 * index
        grid[body[0]] = Cell.HEAD.value + 10 * index
        grid[body[-1]] = Cell.TAIL.value + 10 * index
        snake = Snake(index, body)
        snake.alive = True
        snakes.append(snake)
    base.grid = grid
    base.snakes = snakes
    return base


def view_from(base, head, heading, radius):
    return egocentric_crop(base._padded_rgb(radius), head, heading, radius,
                           radius)


def turn(heading, relative):
    if relative == 1:
        return LEFT_TURN[heading]
    if relative == 2:
        return RIGHT_TURN[heading]
    return heading


def verify(dataset, args, episodes):
    """Reconstructed views must match the stored ones exactly."""
    exact = total = 0
    for index in range(min(episodes, len(dataset))):
        row = dataset[index]
        episode = decode_episode(row)
        env = make_env(row, args)
        env.reset(seed=int(row['seed']))
        radius = int(row['view_radius'])
        for step in range(0, episode['steps'], 5):
            base = apply_state(env, episode['content'][step],
                               episode['body_index'][step])
            for agent in range(episode['num_agents']):
                if not episode['alive_mask'][step, agent]:
                    continue
                pose = episode['poses'][step, agent]
                got = view_from(base, (int(pose[0]), int(pose[1])),
                                HEADINGS[int(pose[2])], radius)
                total += 1
                exact += int(np.array_equal(
                    got, episode['observations'][step, agent]))
    return exact, total


def patch_episode(row, args):
    """Return the row with aftermath frames filled in."""
    episode = decode_episode(row)
    env = make_env(row, args)
    env.reset(seed=int(row['seed']))
    radius = int(row['view_radius'])
    frames = episode['steps'] + 1
    patched = 0

    for agent in range(episode['num_agents']):
        living = np.flatnonzero(episode['alive_mask'][:, agent])
        if len(living) == 0:
            continue
        last = int(living[-1])
        if last + 1 >= frames:
            continue

        pose = episode['poses'][last, agent]
        heading = HEADINGS[int(pose[2])]
        relative = int(episode['ego_actions'][last, agent].argmax())
        moved = turn(heading, relative)
        head = (int(pose[0]) + moved.value[0], int(pose[1]) + moved.value[1])

        base = apply_state(env, episode['content'][last + 1],
                           episode['body_index'][last + 1])
        episode['observations'][last + 1, agent] = view_from(base, head,
                                                             moved, radius)
        episode['poses'][last + 1, agent] = (head[0], head[1],
                                             HEADINGS.index(moved))
        episode['cardinal_actions'][last, agent] = 0
        episode['cardinal_actions'][last, agent,
                                    HEADINGS.index(moved)] = 1
        patched += 1
    return episode, patched


def main():
    args = parse_args()
    for name in args.components:
        source = os.path.join(args.data_root, name)
        dataset = load_from_disk(source)
        exact, total = verify(dataset, args, args.check)
        print(f'{name}: reconstruction check {exact}/{total} exact')
        if exact != total:
            raise SystemExit('reconstruction is not faithful; refusing to '
                             'patch')

        rows, patched = [], 0
        for row in dataset:
            episode, count = patch_episode(row, args)
            rows.append(episode)
            patched += count
        target = os.path.join(args.data_root, name + args.suffix)
        build_dataset(rows).save_to_disk(target + '.tmp')
        if args.suffix == '':
            import shutil
            shutil.rmtree(source)
        os.rename(target + '.tmp', target)
        print(f'  patched {patched} deaths across {len(rows)} episodes '
              f'-> {target}')


if __name__ == '__main__':
    main()
