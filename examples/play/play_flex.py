"""Play the flex world action model, or watch it play itself.

    python examples/play/play_flex.py --model <checkpoint>
    python examples/play/play_flex.py --model <checkpoint> --autonomous

You steer one snake; the others' actions are sampled from the model. The
same three views as play_wam -- the frame the model predicts for the agent
you steer, those frames stitched onto one map, and the real simulator under
the same joint actions.

An older checkpoint plays here identically: one recording no attention
schedule is read as all-global, which is what it was trained as.

Keys
    arrows / WASD   steer          space   pause
    TAB             raw <-> palette-snapped view
    G               save a gif      R      restart      ESC quit
"""
import argparse
import time

import numpy as np
import torch

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.palette import snap_to_palette
from marlenv.core.snake import Direction
from marlenv.flex_wm.model import load_flex_model
from marlenv.flex_wm.runner import (CachedFlexRunner,
                                    FlexRunner)
from marlenv.grading.compare import PALETTE_SNAKES, unrotate_view
from marlenv.wm.canvas import CanvasIntegrator, make_pose
from marlenv.wm.data import to_model_input, to_pixels
from marlenv.wm.interactive import HEADINGS, OPPOSITE
from marlenv.wm.showreel import BACKDROP, INK, MUTED, REWARD_DICT, save

CAPTION = 18


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='marlenv/demodata/flex_wam/model.pt')
    p.add_argument('--schedule', default=None)
    p.add_argument('--autonomous', action='store_true')
    p.add_argument('--agent', type=int, default=0)
    p.add_argument('--immortal-player', action='store_true')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--bootstrap', type=int, default=12)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--bootstrap-sims', type=int, default=48)
    p.add_argument('--rollout-depth', type=int, default=10)
    p.add_argument('--denoise-steps', type=int, default=12)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--window', type=int, default=None)
    p.add_argument('--death-patience', type=int, default=3)
    p.add_argument('--decay', type=float, default=0.95)
    p.add_argument('--tick-ms', type=int, default=400)
    p.add_argument('--scale', type=int, default=34)
    p.add_argument('--canvas-scale', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--num-agents', type=int, default=3)
    p.add_argument('--snake-colors', type=int, default=None)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--record', default=None)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--no-cache', dest='use_cache',
                   action='store_false', default=True,
                   help='recompute the window every pass instead of\n'
                        'encoding each committed step once')
    p.add_argument('--device', default=None)
    return p.parse_args()


class Session:
    """A flex rollout beside the real simulator."""

    def __init__(self, args, model, window, device, seed):
        self.args, self.device, self.agent = args, device, args.agent
        self.env = gym.make(
            'Snake-v1', height=args.side, width=args.side,
            num_snakes=args.num_agents, num_fruits=args.num_fruits,
            reward_dict=REWARD_DICT, view_radius=args.view_radius,
            observation_noise=2.0, snake_noise_sigma=8.0,
            background_gradient=16.0,
            obstacle_density=args.obstacle_density,
            snake_colors=args.snake_colors, disable_env_checker=True)
        self.env.reset(seed=seed)
        base = self.env.unwrapped

        heads = np.array([s.head_coord for s in base.snakes], dtype=np.int64)
        immortal = [args.agent] if args.immortal_player else None
        runner_class = (CachedFlexRunner if args.use_cache
                        else FlexRunner)
        self.runner = runner_class(
            model, agents=list(range(args.num_agents)),
            positions=heads - heads[0], window=window, device=device,
            death_patience=args.death_patience, immortal=immortal)
        self.runner.reset(torch.from_numpy(
            to_model_input(self.observe()[None, None])).to(device))

        self.canvas = CanvasIntegrator(args.side, args.side,
                                       args.view_radius, decay=args.decay)
        self.headings = [s.direction for s in base.snakes]
        self.poses = [make_pose(s.head_coord[0], s.head_coord[1], s.direction)
                      for s in base.snakes]
        self.last_seen = list(self.observe())
        self.paint(self.observe())
        self.alive, self.steps = True, 0
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def observe(self):
        base = self.env.unwrapped
        return np.stack([unrotate_view(view, snake.direction) for view, snake
                         in zip(base.egocentric_rgb(), base.snakes)])

    @property
    def living(self):
        alive = self.runner.alive[0, -1]
        return [i for i in range(len(self.poses)) if bool(alive[i])]

    def dream(self, agent=None):
        frames = self.runner.frames[0, -1]
        if agent is not None:
            return to_pixels(frames[agent].cpu().numpy())
        return [to_pixels(frame.cpu().numpy()) for frame in frames]

    def views_for_display(self):
        living = set(self.living)
        return [view if i in living else self.last_seen[i]
                for i, view in enumerate(self.dream())]

    def paint(self, views):
        self.canvas.fade()
        order = [i for i in self.living if i != self.agent]
        if self.agent in self.living:
            order.append(self.agent)
        for index in order:
            self.canvas.paste(views[index], self.poses[index])
        for index in self.living:
            self.last_seen[index] = views[index]

    def prefill(self, steps, solver):
        base = self.env.unwrapped
        for _ in range(steps):
            if not self.alive:
                break
            _, _, terminated, truncated, _ = self.env.step(
                solver.solve(self.env))
            self.alive = not (all(terminated) or all(truncated))
            self.headings = [s.direction for s in base.snakes]
            self.poses = [make_pose(s.head_coord[0], s.head_coord[1],
                                    s.direction) for s in base.snakes]
            views = self.observe()
            live = torch.tensor([s.alive for s in base.snakes],
                                dtype=torch.bool, device=self.device)
            actions = torch.tensor([HEADINGS.index(h) for h in self.headings],
                                   dtype=torch.long, device=self.device)
            self.runner.observe(actions, torch.from_numpy(
                to_model_input(views[None, None])).to(self.device), live)
            self.paint(views)
            self.steps += 1

    def step(self, cardinal):
        moving = self.living
        fixed = None
        if not self.args.autonomous and cardinal is not None:
            heading = self.headings[self.agent]
            if cardinal is OPPOSITE[heading]:
                cardinal = heading
            fixed = {self.agent: HEADINGS.index(cardinal)}

        actions, _ = self.runner.step(
            fixed=fixed, denoise_steps=self.args.denoise_steps,
            action_steps=self.args.action_steps, generator=self.generator)
        chosen = [HEADINGS[int(a)] for a in actions]
        self.headings = chosen
        for index in moving:
            pose, heading = self.poses[index], chosen[index]
            self.poses[index] = make_pose(pose.row + heading.value[0],
                                          pose.col + heading.value[1],
                                          heading)
        self.paint(self.dream())

        if self.alive:
            base = self.env.unwrapped
            from marlenv.wm.interactive import cardinal_to_ego
            relative = []
            for index, snake in enumerate(base.snakes):
                move = cardinal_to_ego(snake.direction, chosen[index])
                relative.append(0 if move is None else move)
            _, _, term, trunc, _ = self.env.step(relative)
            self.alive = not (all(term) or all(trunc))
        self.steps += 1
        return actions


def compose(session, args, snapped):
    """Dream, canvas, and the simulator beside them."""
    view = session.dream(session.agent)
    if snapped:
        view = snap_to_palette(view, PALETTE_SNAKES)
    up = lambda image, factor: np.repeat(np.repeat(image, factor, 0),
                                         factor, 1)
    panels = [('dream', up(view, args.scale)),
              ('canvas', up(session.canvas.image, args.canvas_scale))]
    if session.alive:
        real = session.observe()[session.agent]
        if snapped:
            real = snap_to_palette(real, PALETTE_SNAKES)
        panels.append(('sim', up(real, args.scale)))

    gap = 12
    height = max(image.shape[0] for _, image in panels)
    width = sum(image.shape[1] for _, image in panels) + gap * (len(panels)
                                                                - 1)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    sheet[:] = BACKDROP
    x = 0
    for _, image in panels:
        sheet[:image.shape[0], x:x + image.shape[1]] = image
        x += image.shape[1] + gap
    return sheet, [name for name, _ in panels]


def build_solver(args, num_actions):
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


def run_headless(session, args):
    frames, start = [], time.time()
    for _ in range(args.steps):
        session.step(None if args.autonomous
                     else session.headings[session.agent])
        frames.append(compose(session, args, True)[0])
        if not session.living:
            break
    elapsed = time.time() - start
    print(f'ran {len(frames)} steps in {elapsed:.1f}s '
          f'({elapsed / max(len(frames), 1) * 1000:.0f} ms/step), '
          f'canvas coverage {session.canvas.coverage():.2f}')
    return frames


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, state = load_flex_model(args.model, device, args.schedule)
    window = args.window or state.get('window') or state.get('context', 48)

    session = Session(args, model, window, device, args.seed)
    if args.bootstrap > 1:
        session.prefill(args.bootstrap - 1,
                        build_solver(args, len(
                            session.env.unwrapped.action_dict)))

    who = ('autonomous' if args.autonomous
           else f'you steer agent {args.agent}')
    print(f'agents: {args.num_agents}   schedule '
          f'{"".join(model.schedule)}   window: {window}   {who}')

    if args.headless:
        frames = run_headless(session, args)
        if args.record:
            save(frames, args.record)
        return

    import pygame
    pygame.init()
    pygame.display.set_caption('marlenv flex world action model')
    font = pygame.font.SysFont('monospace', 15)
    sheet, names = compose(session, args, True)
    screen = pygame.display.set_mode((sheet.shape[1], sheet.shape[0] + 54))
    clock = pygame.time.Clock()

    keys = {pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
            pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT}

    snapped, paused, running = True, True, True
    frames, last_move = [], 0.0
    pending = session.headings[session.agent]
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
                    save(frames, args.record or 'showcase/flex_play.gif')
                elif event.key == pygame.K_r:
                    session = Session(args, model, window, device,
                                      np.random.randint(1 << 30))
                    frames = []

        now = time.time()
        if not paused and (now - last_move) * 1000 >= args.tick_ms:
            # a snake carries on the way it was going; clearing this would
            # leave the model sampling the player's action on every tick
            session.step(pending)
            last_move = now

        sheet, names = compose(session, args, snapped)
        frames.append(sheet)
        surface = pygame.surfarray.make_surface(sheet.swapaxes(0, 1))
        screen.fill(BACKDROP)
        screen.blit(surface, (0, 0))
        status = (f'step {session.steps}   live {len(session.living)}'
                  f'/{args.num_agents}   '
                  f'coverage {session.canvas.coverage():.2f}   '
                  f'{"PAUSED" if paused else f"{clock.get_fps():.0f} fps"}')
        screen.blit(font.render(status, True, INK), (8, sheet.shape[0] + 8))
        screen.blit(font.render('  '.join(names), True, MUTED),
                    (8, sheet.shape[0] + 30))
        pygame.display.flip()
        clock.tick(60)

    if args.record:
        save(frames, args.record)
    pygame.quit()


if __name__ == '__main__':
    main()
