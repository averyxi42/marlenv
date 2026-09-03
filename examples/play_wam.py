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
    p.add_argument('--bootstrap', type=int, default=1,
                   help='frames of real, searched play fed in before the '
                        'model takes over; 1 hands over immediately')
    p.add_argument('--checkpoint', default=None,
                   help='AlphaZero network guiding the bootstrap search; '
                        'without one the search uses random rollouts')
    p.add_argument('--bootstrap-sims', type=int, default=48,
                   help='simulations per bootstrap move')
    p.add_argument('--rollout-depth', type=int, default=10)
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


def build_solver(args, num_actions):
    """The search that plays the bootstrap prefix.

    The prefix is what the model continues from, so it is also a way to ask
    for a particular kind of play: a searched prefix elicits searched play,
    where a random one elicits flailing. That matters here because the
    training mixture holds both.
    """
    from marlenv.policies import AlphaZeroSolver, RolloutEvaluator

    if args.checkpoint:
        from marlenv.policies import NetworkEvaluator, SnakeNet
        state = torch.load(args.checkpoint, map_location='cpu',
                           weights_only=False)
        net = SnakeNet(channels=state.get('channels', 32),
                       blocks=state.get('blocks', 2))
        net.load_state_dict(state['model'])
        evaluator = NetworkEvaluator(net, device=args.device)
    else:
        evaluator = RolloutEvaluator(num_actions,
                                     rollout_depth=args.rollout_depth,
                                     seed=args.seed)
    return AlphaZeroSolver(evaluator, objective='sum',
                           num_simulations=args.bootstrap_sims,
                           max_depth=6, max_joint_actions=32,
                           exploration_fraction=0.0, seed=args.seed)


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

        self.canvas = CanvasIntegrator(args.side, args.side,
                                       args.view_radius, decay=args.decay)
        self.poses = [make_pose(s.head_coord[0], s.head_coord[1], s.direction)
                      for s in base.snakes]
        self.paint(views)
        self.alive = True
        self.steps = 0
        self.generator = torch.Generator(device=device).manual_seed(seed)
        if args.bootstrap > 1:
            self.prefill(args.bootstrap - 1)

    def observe(self):
        """Every agent's view, north-up."""
        base = self.env.unwrapped
        return np.stack([unrotate_view(v, s.direction) for v, s
                         in zip(base.egocentric_rgb(), base.snakes)])

    def prefill(self, steps):
        """Play real steps into the context before the model takes over.

        The frames and actions are the simulator's, but they enter the
        context by the same route a generated step does, so the model
        cannot tell where the prefix ends. Two things come of it: a longer
        history to condition on, which steadies what follows, and control
        over which policy the rollout continues.

        The canvas is painted as this runs, so what it shows is what the
        model actually has in context rather than only the dreamt part.
        """
        base = self.env.unwrapped
        solver = build_solver(self.args, len(base.action_dict))
        for _ in range(steps):
            if not self.alive:
                break
            _, _, terminated, truncated, _ = self.env.step(solver.solve(
                self.env))
            self.alive = not (all(terminated) or all(truncated))

            # read the poses back rather than dead reckoning them: after a
            # death the simulator's head is already the aftermath viewpoint,
            # which is exactly what the observation is centred on
            self.headings = [snake.direction for snake in base.snakes]
            self.poses = [make_pose(snake.head_coord[0], snake.head_coord[1],
                                    snake.direction)
                          for snake in base.snakes]
            live = torch.tensor([snake.alive for snake in base.snakes],
                                dtype=torch.bool, device=self.device)
            actions = torch.tensor([HEADINGS.index(h) for h in self.headings],
                                   dtype=torch.long, device=self.device)

            views = self.observe()
            frame = torch.from_numpy(
                to_model_input(views[None, None])).to(self.device)
            self.runner.observe(actions, frame, live)
            self.paint(views)
            self.steps += 1

    def dream(self, agent=None):
        """The frames the model just generated, as north-up pixels.

        With an agent index, that agent's view; without one, every agent's.
        """
        frames = self.runner.frames[0, -1]
        if agent is not None:
            return to_pixels(frames[agent].cpu().numpy())
        return [to_pixels(frame.cpu().numpy()) for frame in frames]

    @property
    def pose(self):
        """The played agent's pose, for the head marker and the status line."""
        return self.poses[self.agent]

    @property
    def living(self):
        """Indices of the viewpoints still being updated."""
        alive = self.runner.alive[0, -1]
        return [i for i in range(len(self.poses)) if bool(alive[i])]

    def paint(self, views):
        """Composite every living agent's view onto the shared canvas.

        All the agents look at the same board, so all of their views belong
        on it; painting only the played one threw away most of what the model
        generates each step, and left the canvas as sparse as a single-agent
        rollout. Dead viewpoints are skipped -- the runner keeps a black
        placeholder in their slot, which is not a view of anything.

        Ageing happens once for the whole step rather than once per view, or
        the canvas would fade N times faster with N agents. The played agent
        goes last so its own view wins wherever the views overlap.
        """
        self.canvas.fade()
        order = [i for i in self.living if i != self.agent]
        if self.agent in self.living:
            order.append(self.agent)
        for index in order:
            self.canvas.paste(views[index], self.poses[index])

    def step(self, cardinal):
        # captured before the step: a viewpoint that dies this step still
        # advances into the cell it died entering, and freezes after that
        moving = self.living
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
        for index in moving:
            heading = chosen[index]
            pose = self.poses[index]
            self.poses[index] = make_pose(pose.row + heading.value[0],
                                          pose.col + heading.value[1], heading)
        self.paint(self.dream())

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
    view = session.dream(session.agent)
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

    pending, snapped, paused, running = None, True, True, True
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
