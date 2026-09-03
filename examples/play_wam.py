"""Play the multi-agent world action model, or watch it play itself.

    python examples/play_wam.py --model marlenv/demodata/wm_multi/model.pt
    python examples/play_wam.py --model ... --autonomous --steps 200

You steer one snake; the others' actions are *sampled from the model*,
which is what a world action model buys -- a plain world model cannot be
rolled out with several agents because the others' actions are policies,
not inputs. With --autonomous nobody steers and every action is sampled.

Panels, the same three as the single-agent player
    dream     the frame the model predicts for the agent you steer
    canvas    those frames stitched onto one map at their dead reckoned
              positions, fading with age
    sim       the real simulator under the same joint actions, so the two
              can be compared step for step

Keys
    arrows / WASD   steer          space   pause
    TAB             raw <-> palette-snapped view
    G               save a gif      R      restart      ESC quit
"""
import argparse
import os
import time

import numpy as np
import torch

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.palette import snap_to_palette
from marlenv.core.snake import Direction
from marlenv.grading.compare import PALETTE_SNAKES, unrotate_view
from marlenv.wm.canvas import CanvasIntegrator, make_pose
from marlenv.wm.data import to_model_input, to_pixels
from marlenv.wm.interactive import HEADINGS, OPPOSITE
from marlenv.wm.marunner import CachedMultiRunner, MultiAgentRunner
from marlenv.wm.multiagent import MultiAgentWorldModel

REWARD_DICT = {'fruit': 1.0, 'kill': 0.0, 'lose': -5.0, 'win': 0.0,
               'time': 0.01}
BACKDROP = (18, 18, 22)
INK = (232, 232, 238)
MUTED = (140, 140, 150)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='marlenv/demodata/wm_multi/model.pt')
    p.add_argument('--autonomous', action='store_true',
                   help='nobody steers; every action is sampled')
    p.add_argument('--agent', type=int, default=0, help='the one you steer')
    p.add_argument('--steps', type=int, default=120,
                   help='length of an autonomous or demo run')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--denoise-steps', type=int, default=12)
    p.add_argument('--action-steps', type=int, default=6)
    p.add_argument('--window', type=int, default=None)
    p.add_argument('--no-cache', dest='use_cache', action='store_false',
                   default=True,
                   help='recompute the window every denoising pass instead '
                        'of using the KV cache')
    p.add_argument('--decay', type=float, default=0.95)
    p.add_argument('--tick-ms', type=int, default=140)
    p.add_argument('--scale', type=int, default=34)
    p.add_argument('--canvas-scale', type=int, default=16)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--record', default=None)
    p.add_argument('--headless', action='store_true',
                   help='run without a window, for checking or recording')
    p.add_argument('--device', default=None)
    return p.parse_args()


def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = MultiAgentWorldModel(
        num_agents=state['num_agents'], view=state.get('view', 9),
        num_actions=state.get('num_actions', 4), frame='world',
        dim=state['dim'], depth=state['depth'], heads=state['heads'])
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state.get('context', 24)


class Session:
    """A world action rollout beside the real simulator."""

    def __init__(self, args, model, context, device, seed):
        self.args = args
        self.device = device
        self.agent = args.agent
        self.env = gym.make(
            'Snake-v1', height=args.side, width=args.side,
            num_snakes=model.num_agents, num_fruits=args.num_fruits,
            reward_dict=REWARD_DICT, view_radius=args.view_radius,
            observation_noise=2.0, snake_noise_sigma=8.0,
            background_gradient=16.0, obstacle_density=args.obstacle_density,
            disable_env_checker=True)
        self.env.reset(seed=seed)
        base = self.env.unwrapped

        views = self.observe()
        heads = np.array([s.head_coord for s in base.snakes], dtype=np.int64)
        origins = torch.from_numpy(heads - heads[0])[None]

        runner_class = (CachedMultiRunner if args.use_cache
                        else MultiAgentRunner)
        self.runner = runner_class(model, origins,
                                   window=args.window or context,
                                   device=device)
        # (batch, time, agents, view, view, channels)
        self.runner.reset(torch.from_numpy(
            to_model_input(views[None, None])).to(device))
        self.headings = [s.direction for s in base.snakes]

        snake = base.snakes[self.agent]
        self.canvas = CanvasIntegrator(args.side, args.side,
                                       args.view_radius, decay=args.decay)
        self.pose = make_pose(snake.head_coord[0], snake.head_coord[1],
                              snake.direction)
        self.canvas.add(views[self.agent], self.pose)
        self.alive = True
        self.steps = 0
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def observe(self):
        """Every agent's view, north-up."""
        base = self.env.unwrapped
        return np.stack([unrotate_view(v, s.direction) for v, s
                         in zip(base.egocentric_rgb(), base.snakes)])

    def dream(self):
        return to_pixels(self.runner.frames[0, -1, self.agent].cpu().numpy())

    def step(self, cardinal):
        fixed = None
        if not self.args.autonomous and cardinal is not None:
            heading = self.headings[self.agent]
            if cardinal is OPPOSITE[heading]:
                cardinal = heading
            fixed = {self.agent: HEADINGS.index(cardinal)}

        actions, _ = self.runner.step(fixed=fixed,
                                      denoise_steps=self.args.denoise_steps,
                                      action_steps=self.args.action_steps,
                                      generator=self.generator)
        chosen = [HEADINGS[int(a)] for a in actions]
        self.headings = chosen
        self.pose = make_pose(self.pose.row + chosen[self.agent].value[0],
                              self.pose.col + chosen[self.agent].value[1],
                              chosen[self.agent])
        self.canvas.add(self.dream(), self.pose)

        # the simulator takes the same joint action, converted to relative
        if self.alive:
            base = self.env.unwrapped
            relative = []
            for index, snake in enumerate(base.snakes):
                want = chosen[index]
                from marlenv.wm.interactive import cardinal_to_ego
                move = cardinal_to_ego(snake.direction, want)
                relative.append(0 if move is None else move)
            _, _, term, trunc, _ = self.env.step(relative)
            self.alive = not (all(term) or all(trunc))
        self.steps += 1
        return actions


def compose(session, args, snapped):
    view = session.dream()
    if snapped:
        view = snap_to_palette(view, PALETTE_SNAKES)
    panels = [np.repeat(np.repeat(view, args.scale, 0), args.scale, 1)]
    canvas = session.canvas.image
    panels.append(np.repeat(np.repeat(canvas, args.canvas_scale, 0),
                            args.canvas_scale, 1))
    names = ['dream', 'canvas']
    if session.alive:
        real = session.observe()[session.agent]
        if snapped:
            real = snap_to_palette(real, PALETTE_SNAKES)
        panels.append(np.repeat(np.repeat(real, args.scale, 0),
                                args.scale, 1))
        names.append('sim')

    gap = 12
    height = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    sheet[:] = BACKDROP
    x = 0
    for panel in panels:
        sheet[:panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return sheet, names


def save_gif(frames, path, duration=140):
    from PIL import Image
    if not frames:
        print('nothing to save')
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   format='GIF', loop=0, duration=duration)
    print(f'wrote {len(images)} frames to {path}')


def run_headless(session, args):
    frames = []
    start = time.time()
    for _ in range(args.steps):
        session.step(None if args.autonomous else Direction.UP)
        frames.append(compose(session, args, True)[0])
    elapsed = time.time() - start
    print(f'ran {args.steps} steps in {elapsed:.1f}s '
          f'({elapsed / args.steps * 1000:.0f} ms/step), '
          f'canvas coverage {session.canvas.coverage():.2f}')
    return frames


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, context = load_model(args.model, device)
    who = ('autonomous' if args.autonomous
           else f'you steer agent {args.agent}')
    print(f'agents: {model.num_agents}   context: {context}   {who}')

    session = Session(args, model, context, device, args.seed)

    if args.headless:
        frames = run_headless(session, args)
        if args.record:
            save_gif(frames, args.record)
        return

    import pygame
    pygame.init()
    pygame.display.set_caption('marlenv world action model')
    font = pygame.font.SysFont('monospace', 15)
    sheet, names = compose(session, args, True)
    screen = pygame.display.set_mode((sheet.shape[1], sheet.shape[0] + 54))
    clock = pygame.time.Clock()

    keys = {pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
            pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT}

    pending, snapped, paused, running = None, True, False, True
    frames, last_move = [], 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in keys:
                    pending = keys[event.key]
                elif event.key == pygame.K_TAB:
                    snapped = not snapped
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_g:
                    save_gif(frames, args.record or 'showcase/wam.gif')
                elif event.key == pygame.K_r:
                    session = Session(args, model, context, device,
                                      np.random.randint(1 << 30))
                    frames = []

        now = time.time()
        if not paused and (now - last_move) * 1000 >= args.tick_ms:
            session.step(pending)
            pending = None
            last_move = now

        sheet, names = compose(session, args, snapped)
        frames.append(sheet)
        screen.fill(BACKDROP)
        screen.blit(pygame.surfarray.make_surface(sheet.swapaxes(0, 1)),
                    (0, 0))
        facing = session.headings[session.agent].name.lower()
        mode = 'autonomous' if args.autonomous else f'facing {facing}'
        rate = 'PAUSED' if paused else f'{clock.get_fps():.0f} fps'
        status = (f'step {session.steps}   {mode}   '
                  f'coverage {session.canvas.coverage():.2f}   {rate}')
        screen.blit(font.render(status, True, INK), (8, sheet.shape[0] + 8))
        screen.blit(font.render('  '.join(names), True, MUTED),
                    (8, sheet.shape[0] + 30))
        pygame.display.flip()
        clock.tick(60)

    if args.record:
        save_gif(frames, args.record)
    pygame.quit()


if __name__ == '__main__':
    main()
