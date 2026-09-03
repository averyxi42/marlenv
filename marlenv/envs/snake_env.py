from collections import defaultdict, deque
from typing import List, Tuple

import os
import math
import datetime
import warnings

import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding

from marlenv.core.grid_util import (
    random_empty_coords, draw, make_grid, dfs_sweep_empty,
    rgb_from_grid, image_from_grid, image_from_rgb, add_obstacles,
    MEAN_OBSTACLE_AREA)
from marlenv.core.noise import ObservationNoise
from marlenv.core.observation import egocentric_crop, pad_grid
from marlenv.core.render import draw_frame, image_from_env
from marlenv.core.snake import Direction, Snake, Cell, CellColors
from marlenv.envs.constants import FEATURE_CHANNEL, RGB_CHANNEL


class SnakeEnv(gym.Env):
    default_action_dict = {
        'noop': 0,
        'left': 1,
        'right': 2,
        'down': 3,
        'up': 4
    }

    action_angle_dict = {
        0: 0.0,
        1: math.pi / 2.0,
        2: -math.pi / 2.0
    }

    default_reward_dict = {
        'fruit': 1.0,
        'kill': 0.,
        'lose': 0.0,
        'win': 0.,
        'time': 0.,
    }
    reward_keys = default_reward_dict.keys()

    max_episode_steps = 1e4

    def __init__(
            self,
            height=20,
            width=20,
            num_snakes=4,
            snake_length=3,
            vision_range=None,
            frame_stack=1,
            observer='snake',
            # if 'snake' three actions, if 'human' five actions
            *args,
            **kwargs
    ):
        """
        kwargs
        'reward_dict', 'num_fruits'
        """

        reward_dict = kwargs.pop('reward_dict', SnakeEnv.default_reward_dict)
        if reward_dict.keys() != SnakeEnv.reward_keys:
            raise KeyError(
                f'reward dict keys must correspond to {SnakeEnv.reward_keys}'
            )
        else:
            self.reward_dict = reward_dict
        self.max_episode_steps = kwargs.pop('max_episode_steps',
                                            SnakeEnv.max_episode_steps)
        # 'classic' keeps the original flat colouring; 'pixel' uses the
        # retro renderer, which also shows each segment's direction
        self.render_style = kwargs.pop('render_style', 'classic')
        self.cell_size = kwargs.pop('cell_size', 16)

        # procedural layout. Both default to off, so an env constructed the
        # old way still gets a fixed, empty board.
        # RGB observation noise, bound to cells and to body position so it
        # is fixed for the episode rather than resampled per frame
        self.observation_noise = kwargs.pop('observation_noise', 0.0)
        # snake noise sits on saturated colours rather than dark cells, so
        # it takes its own sigma; None follows observation_noise
        self.snake_noise_sigma = kwargs.pop('snake_noise_sigma', None)
        self.noise_period = kwargs.pop('noise_period', 8)
        self.obs_noise = None

        # radius of the egocentric RGB view; the view is
        # (2 * view_radius + 1) square. None disables it. Distinct from
        # vision_range, which crops the learned feature planes.
        self.view_radius = kwargs.pop('view_radius', None)

        self.grid_size_range = kwargs.pop('grid_size_range', None)
        # obstacle_density is the fraction of interior cells walled off;
        # num_obstacles overrides it with an exact piece count
        self.obstacle_density = kwargs.pop('obstacle_density', 0.0)
        self.num_obstacles = kwargs.pop('num_obstacles', None)

        self.num_snakes = num_snakes
        self.num_fruits = kwargs.pop('num_fruits',
                                     int(round(num_snakes * 0.8)))
        self.grid_shape: Tuple = (height, width)
        self.grid: np.ndarray
        self.snakes: List[Snake]
        self.snake_length = snake_length
        self.vision_range = vision_range
        self.observer = observer

        self.low = 0
        self.image_obs = False
        if self.image_obs:
            self.high = 255
        else:
            self.high = 1
        if self.observer == 'human':
            self.action_dict = SnakeEnv.default_action_dict
        elif self.observer == 'snake':
            self.action_dict = SnakeEnv.action_angle_dict

        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.action_dict)] * self.num_snakes
        )

        self.frame_stack = frame_stack
        default_ch = RGB_CHANNEL if self.image_obs else FEATURE_CHANNEL
        self.obs_ch = default_ch * self.frame_stack

        if self.vision_range:
            h = w = self.vision_range * 2 + 1
            self.observation_space = gym.spaces.Box(
                self.low,
                self.high,
                shape=[self.num_snakes, h, w, self.obs_ch],
                dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                self.low,
                self.high,
                shape=[self.num_snakes, *self.grid_shape, self.obs_ch],
                dtype=np.uint8
            )

    def reset(self, *, seed=None, options=None):
        # seeds self.np_random, which drives every random draw below
        super().reset(seed=seed)
        if self.grid_size_range is not None:
            self._resample_grid_shape()
        # Create the grid base, then scatter any obstacles into it before
        # snakes and fruit are placed, so both avoid them for free
        self.grid = make_grid(*self.grid_shape,
                              empty_value=Cell.EMPTY.value,
                              wall_value=Cell.WALL.value)
        self._place_obstacles()
        # Generate and add snake to the grid
        self.snakes = self._generate_snakes()
        for snake in self.snakes:
            coords = snake.coords
            snake_id = 10 * snake.idx
            for coord in coords:
                self.grid[coord] = Cell.BODY.value + snake_id
            self.grid[snake.head_coord] = Cell.HEAD.value + snake_id
            self.grid[snake.tail_coord] = Cell.TAIL.value + snake_id

        # Generate fruit and add to the grid
        xs, ys = self._generate_fruits(self.num_fruits)
        self.grid[xs, ys] = Cell.FRUIT.value

        self.alive_snakes = self.num_snakes
        self.frame_buffer = []

        obs = self._init_obs()

        # Drawn from a spawned child stream, which leaves the env's own
        # generator untouched: an episode plays out identically whether or
        # not noise is enabled, so noisy and clean renders are paired
        if self.observation_noise or self.snake_noise_sigma:
            self.obs_noise = ObservationNoise(
                self.grid_shape, self.num_snakes,
                sigma=self.observation_noise,
                snake_sigma=self.snake_noise_sigma,
                period=self.noise_period,
                pad=self.view_radius or 0,
                np_random=self.np_random.spawn(1)[0])

        # Episodic stats
        self._reset_epi_stats()
        self.episode_length = 0

        return np.array(obs, dtype=np.uint8), {}

    def _resample_grid_shape(self):
        """Pick a new square board size for this episode.

        Square because the learned observation canonicalises headings by
        rotating the grid, which only preserves shape on a square board.
        """
        low, high = self.grid_size_range
        side = int(self.np_random.integers(low, high + 1))
        if self.grid_shape != (side, side):
            self.grid_shape = (side, side)
            self._rebuild_observation_space()

    def _place_obstacles(self):
        """Scatter interior walls, leaving room for every snake to spawn.

        The count is retried downwards rather than the layout being
        regenerated, so a board that is too crowded degrades to a sparser
        one instead of failing.
        """
        count = self.num_obstacles
        if count is None:
            # obstacle_density is the target fraction of interior cells that
            # become wall; pieces cover several cells each
            interior = (self.grid_shape[0] - 2) * (self.grid_shape[1] - 2)
            count = int(round(interior * self.obstacle_density
                              / MEAN_OBSTACLE_AREA))
        if count <= 0:
            return

        clean = self.grid.copy()
        while count > 0:
            self.grid[:] = clean
            add_obstacles(self.grid, count, np_random=self.np_random,
                          wall_value=Cell.WALL.value,
                          empty_value=Cell.EMPTY.value)
            if len(dfs_sweep_empty(self.grid, self.snake_length)) >= \
                    4 * self.num_snakes:
                return
            count //= 2
        self.grid[:] = clean

    def _rebuild_observation_space(self):
        shape = ([self.num_snakes, self.vision_range * 2 + 1,
                  self.vision_range * 2 + 1, self.obs_ch]
                 if self.vision_range
                 else [self.num_snakes, *self.grid_shape, self.obs_ch])
        self.observation_space = gym.spaces.Box(
            self.low, self.high, shape=shape, dtype=np.uint8)

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='ascii', style=None):
        style = style or self.render_style
        if mode == 'ascii':
            # Just for debugging
            SYM2CHR = {
                Cell.EMPTY.value: '.',
                Cell.WALL.value: '#',
                Cell.FRUIT.value: 'o',
                Cell.BODY.value: 'b',
                Cell.HEAD.value: 'H',
                Cell.TAIL.value: 't'
            }

            # CHR2SYM = {v: k for k, v in SYM2CHR.items()}

            data = np.vectorize(SYM2CHR.get)(self.grid % 10)
            data = [''.join(i) for i in data]
            data = '\n'.join(data)
            print(data)
        elif mode == 'gif':
            # Save game play as gif
            # Create a game scene from the grid
            # Append to buffer
            if style == 'pixel':
                game_frame = image_from_env(self.grid, self.snakes,
                                            self.cell_size)
            else:
                game_frame = image_from_rgb(self._classic_rgb())
            self.frame_buffer.append(game_frame)
        elif mode == 'rgb_array':
            if style == 'pixel':
                return draw_frame(self.grid, self.snakes, self.cell_size)
            return self._classic_rgb()
        elif mode == 'human':
            # Run pygame
            pass

    def _classic_rgb(self):
        """One pixel per cell, plus this episode's bound noise if enabled."""
        rgb_array = rgb_from_grid(self.grid, Cell, CellColors)
        if self.obs_noise is not None:
            rgb_array = self.obs_noise.apply(rgb_array, self.snakes)
        return rgb_array

    def _padded_rgb(self, pad):
        """The board rendered with ``pad`` cells of free space around it."""
        rgb_array = rgb_from_grid(pad_grid(self.grid, pad), Cell, CellColors)
        if self.obs_noise is not None:
            rgb_array = self.obs_noise.apply(rgb_array, self.snakes, pad=pad)
        return rgb_array

    def egocentric_rgb(self):
        """Each snake's own view: ``(num_snakes, size, size, 3)`` uint8.

        Odd-sized, centred on the head and rotated so the snake faces up.
        Dead snakes get an all-zero view, since they observe nothing.
        """
        if not self.view_radius:
            raise RuntimeError(
                'egocentric_rgb() needs view_radius set on the env')
        radius = self.view_radius
        size = 2 * radius + 1
        frame = self._padded_rgb(radius)
        views = np.zeros((self.num_snakes, size, size, 3), dtype=np.uint8)
        for i, snake in enumerate(self.snakes):
            if snake.alive:
                views[i] = egocentric_crop(frame, snake.head_coord,
                                           snake.direction, radius, radius)
        return views

    def close(self):
        pass

    def step(self, actions):
        """
        Time Complexity:
            O(3*num_snakes)
            + (O(H*W) w/ fruits(maybe better approach, like random polling))

        updates snake status and grid
        all snake statuses are updated only in step method
        """

        if isinstance(actions, int):
            actions = [actions]
        assert len(actions) == self.num_snakes
        for i, ac in enumerate(actions):
            if isinstance(ac, np.ndarray):
                actions[i] = ac.item()
        # preprocess
        next_head_coords = defaultdict(list)
        alive_snakes = []
        for snake, action in zip(self.snakes, actions):
            if snake.alive:
                if self.observer == 'human':
                    snake.direction = self._next_direction_global(
                        snake.direction, action)
                elif self.observer == 'snake':
                    snake.direction = self._next_direction(snake.direction,
                                                           action)
                new_head_coord = snake.head_coord + snake.direction
                next_head_coords[new_head_coord].append(snake.idx)
                alive_snakes.append(snake.idx)
        dead_idxes, fruit_idxes, fruit_taken = self._check_collision(
            next_head_coords)

        self.alive_snakes -= len(dead_idxes)
        for idx in dead_idxes:
            self.snakes[idx].death = True
            self.snakes[idx].alive = False
        for idx in fruit_idxes:
            tail_coord = self.snakes[idx].tail_coord
            if tail_coord in next_head_coords.keys():
                for di in next_head_coords[tail_coord]:
                    self.snakes[di].death = True
                    self.snakes[di].alive = False
                    self.alive_snakes -= 1
                    self.snakes[idx].kills += 1
            self.snakes[idx].fruit = True
        if self.alive_snakes == 1 and self.num_snakes > 1:
            for snake in self.snakes:
                if snake.alive:
                    # print('player {} wins!'.format(snake.idx))
                    snake.win = True
                    break

        rews = []
        dones = []
        fruits, kills = [], []
        # Update snake rews, stats, etc., and update the grid accordingly
        for snake in self.snakes:
            if not snake.death and not snake.alive:
                snake.reward = 0.
                rews.append(snake.reward)
                fruits.append(0)
                kills.append(0)
            else:
                snake.reward = self.reward_dict['time'] * snake.alive
                snake.reward += self.reward_dict['fruit'] * snake.fruit
                snake.reward += self.reward_dict['lose'] * snake.death
                snake.reward += self.reward_dict['kill'] * snake.kills
                snake.reward += self.reward_dict['win'] * snake.win
                rews.append(snake.reward)
                fruits.append(float(snake.fruit))
                kills.append(float(snake.kills))
                self._update_grid(snake)
            dones.append(not snake.alive)

        # Generate fruit and add to the grid
        xs, ys = self._generate_fruits(fruit_taken)
        if xs is not None:
            self.grid[xs, ys] = Cell.FRUIT.value

        obs = self._get_obs()

        # for s_idx, rew in enumerate(rews):
        #     self.epi_scores[s_idx] += rew
        done_mask = 1. - np.asarray(dones)
        self.epi_scores = self.epi_scores + done_mask * np.asarray(rews)
        self.epi_steps = self.epi_steps + done_mask * np.ones(len(dones))
        self.epi_fruits = self.epi_fruits + done_mask * np.asarray(fruits)
        self.epi_kills = self.epi_kills + done_mask * np.asarray(kills)

        info = {}
        self.episode_length += 1
        truncated = [False] * self.num_snakes
        if self.episode_length >= self.max_episode_steps:
            # snakes still alive at the time limit are truncated, not
            # terminated; the ones already dead keep their done flag
            truncated = [not done for done in dones]

        if self._done_fn(dones) or any(truncated):
            sorted_scores = np.unique(np.sort(self.epi_scores)[::-1])
            ranks = np.array([0 for _ in range(self.num_snakes)])
            base_rank = 1
            for score in sorted_scores[::-1]:
                idx = np.where(np.array(self.epi_scores) == score)[0]
                ranks[idx] = base_rank
                base_rank += len(idx)
            info['rank'] = list(ranks)

            info.update(
                {'episode_scores': self.epi_scores,
                 'episode_steps': self.epi_steps,
                 'episode_fruits': self.epi_fruits,
                 'episode_kills': self.epi_kills})

            self._reset_epi_stats()

        return np.array(obs, dtype=np.uint8), rews, dones, truncated, info

    def _done_fn(self, dones):
        return all(dones)

    def save_gif(self, fp=None):
        if fp is None:
            curr_dir = os.getcwd()
            save_dir = os.path.join(curr_dir, 'tmp')
            now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            fp = os.path.join(save_dir, '{}.gif'.format(now))
            os.makedirs(save_dir, exist_ok=True)
        if not self.frame_buffer:
            warnings.warn(
                "You must call render('gif') first. No images to save."
            )
        else:
            print('Saving image to {}'.format(fp))
            self.frame_buffer[0].save(fp, save_all=True,
                                      append_images=self.frame_buffer[1:],
                                      format='GIF',
                                      loop=0)
        return fp

    def _reset_epi_stats(self):
        self.epi_scores = np.zeros(self.num_snakes)
        self.epi_steps = np.zeros(self.num_snakes)
        self.epi_fruits = np.zeros(self.num_snakes)
        self.epi_kills = np.zeros(self.num_snakes)

    def _init_obs(self):
        if self.image_obs:
            self.obs = deque(maxlen=self.frame_stack)
            for _ in range(self.frame_stack):
                self.obs.append(rgb_from_grid(self.grid, Cell, CellColors))
            obs = [np.concatenate(list(self.obs), axis=-1)
                   for _ in range(self.num_snakes)]
        else:
            self.obs = deque(maxlen=self.frame_stack)
            _obs = self._encode(self.grid, vision_range=self.vision_range)
            for _ in range(self.frame_stack):
                self.obs.append(_obs)
            obs = list(zip(*list(self.obs)))
            obs = [np.concatenate(o, axis=-1) for o in obs]

        return obs

    def _get_obs(self):
        if self.image_obs:
            self.obs.append(rgb_from_grid(self.grid, Cell, CellColors))
            obs = [np.concatenate(list(self.obs), axis=-1)
                   for _ in range(self.num_snakes)]
        else:
            _obs = self._encode(self.grid, vision_range=self.vision_range)
            self.obs.append(_obs)
            obs = list(zip(*list(self.obs)))
            obs = [np.concatenate(o, axis=-1) for o in obs]

        return obs

    def _encode(self, obs, vision_range=None):
        # Encode the observation. obs is self.grid
        # Returns the obs in HxWxC
        # May be overriden for customized observation
        env_objs = np.zeros([*obs.shape, 2], dtype=np.float32)
        snake_objs = np.zeros([*obs.shape,
                               3 * 2, self.num_snakes], dtype=np.float32)
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                cell_value = obs[r, c]
                if cell_value in (Cell.WALL.value, Cell.FRUIT.value):
                    env_objs[r, c, cell_value - 1] = 1
                elif cell_value != Cell.EMPTY.value:
                    snake_id = cell_value // 10
                    obj_id = cell_value % 10
                    myself = np.zeros(self.num_snakes)
                    myself[snake_id] = 1
                    snake_objs[r, c, obj_id, :] = myself
                    snake_objs[r, c, obj_id - 3, :] = 1 - myself
        encoded_obs = [np.concatenate([env_objs, snake_objs[..., idx]],
                                      axis=-1)
                       for idx in range(self.num_snakes)]

        if vision_range:
            cropped_encoded_obs = []
            for full_obs in encoded_obs:
                head_pos = np.unravel_index(full_obs[:, :, 5].argmax(),
                                            full_obs[:, :, 5].shape)
                head_pos = np.array(head_pos)
                crop_range_min = np.maximum(head_pos - vision_range, 0)
                crop_range_max = np.minimum(head_pos + vision_range,
                                            np.array(obs.shape) - 1)
                cropped_obs = np.zeros((vision_range * 2 + 1,
                                        vision_range * 2 + 1,
                                        full_obs.shape[-1]))
                start = crop_range_min - head_pos + vision_range
                end = crop_range_max - head_pos + vision_range
                cropped_full_obs = full_obs[
                                   crop_range_min[0]:crop_range_max[0] + 1,
                                   crop_range_min[1]:crop_range_max[1] + 1, :]
                cropped_obs[start[0]:end[0] + 1,
                start[1]:end[1] + 1, :] = cropped_full_obs
                cropped_encoded_obs.append(cropped_obs)
            encoded_obs = cropped_encoded_obs

        return encoded_obs

    def _check_collision(self, next_head_coords):
        # Check for head, body, wall, fruit collision and assign new status
        dead_idxes = []
        fruit_idxes = []
        fruit_taken = 0
        for coord, idxes in next_head_coords.items():
            cell_value = self.grid[coord] % 10
            # Head collision or clear death
            if len(idxes) > 1 or cell_value in (Cell.WALL.value,
                                                Cell.BODY.value,
                                                Cell.HEAD.value):
                # if len(idxes) > 1:
                #     print("Death by collision", idxes)
                dead_idxes.extend(idxes)
                if cell_value == Cell.FRUIT.value:
                    fruit_taken += 1
                if cell_value in (Cell.BODY.value, Cell.HEAD.value):
                    self.snakes[self.grid[coord] // 10].kills += 1
            elif len(idxes) == 1 and cell_value == Cell.FRUIT.value:
                fruit_idxes.extend(idxes)
                fruit_taken += 1
        dead_idxes = list(set(dead_idxes))

        return dead_idxes, fruit_idxes, fruit_taken

    def _update_grid(self, snake):
        # Update the grid according to a snake's status
        if snake.alive:
            snake_id = 10 * snake.idx  # For distinguishing snakes in grid
            self.grid[snake.head_coord] = Cell.BODY.value + snake_id
            # could be current or prev if ate fruit
            prev_tail_coord = snake.move()
            if prev_tail_coord:
                # Didn't eat the fruit
                if self.grid[prev_tail_coord] == Cell.TAIL.value + snake_id:
                    self.grid[prev_tail_coord] = Cell.EMPTY.value
            self.grid[snake.head_coord] = Cell.HEAD.value + snake_id

            self.grid[snake.tail_coord] = Cell.TAIL.value + snake_id
        else:
            coord_list = snake.coords
            if self.grid[snake.coords[-1]] // 10 != snake.idx:
                coord_list = coord_list[:-1]
            if draw(self.grid, coord_list, Cell.EMPTY.value) is False:
                print('draw failed')
            snake.move()

    def _clear_overlap(self, list_of_coords):
        # Check for overlap in given snake coordinates
        flat_list = []
        for element in list_of_coords:
            flat_list.extend(element)
        unique_list = list(set(flat_list))
        return len(unique_list) == len(flat_list)

    def _generate_snakes(self):
        # Depth-first-search through the grid for possible snake positions
        candidates = dfs_sweep_empty(self.grid, self.snake_length)
        for _ in range(1000):
            # Randomly select init snake poses untill no overlap
            sample_idx = self.np_random.permutation(
                len(candidates)
            )[:self.num_snakes]
            samples = [candidates[si] for si in sample_idx]
            if self._clear_overlap(samples):
                break
        else:
            raise RuntimeError(
                f'could not place {self.num_snakes} snakes of length '
                f'{self.snake_length} on this board; it is too crowded'
            )
        snakes = [Snake(idx, coords) for idx, coords in enumerate(samples)]

        return snakes

    def _generate_fruits(self, num_fruits=1):
        xs = ys = None
        if num_fruits:
            xs, ys = random_empty_coords(self.grid, num_coords=num_fruits,
                                         np_random=self.np_random)

        return xs, ys

    def _next_direction(self, direction, action):
        """
        0 == noop
        1 == left
        2 == right
        Change direction by +-90 degrees
        """
        angle = math.atan2(direction.value[1], direction.value[0])
        new_coord = (int(math.cos(angle + self.action_angle_dict[action])),
                     int(math.sin(angle + self.action_angle_dict[action])))
        return Direction(new_coord)

    def _next_direction_global(self, direction, action):
        """
        0 == noop
        1 == left
        2 == right
        3 == down
        4 == up
        Change direction to given global direction
        """
        new_direction = direction
        if direction.value[0] == 0:
            # snake is moving up or down
            if action == 3:
                new_direction = Direction.DOWN
            elif action == 4:
                new_direction = Direction.UP
        elif direction.value[1] == 0:
            # snake is moving lfet or right
            if action == 1:
                new_direction = Direction.LEFT
            elif action == 2:
                new_direction = Direction.RIGHT
        return new_direction