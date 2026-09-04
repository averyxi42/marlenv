# Marlenv

Marlenv is a multi-agent environment for reinforcement learning, based on the OpenAI [gym](https://github.com/openai/gym) convention. 

The function names such as reset(), step() are consistent but the return format is different. Unlike the single agent environments, the multi-agent environments included in this repo formats all returns in a list format, where each element corresponds to each agent in the environment. A similar rule applies to the input action where the action should be a list of actions with a length of number of agents. 

Marlenv is an ongoing project and modifications and new environments are expected in the future. 


## Installation

Three commands, from a clone of the repository:

```bash
conda create -n marlenv python=3.12 -y
conda activate marlenv
pip install -e .
```

That brings up everything the example scripts need -- the environment, the
search policies, the world models, and the data pipeline -- because they
are the project rather than optional extras. `pip install -e '.[dev]'`
adds pytest if you want to run the suite.

On Linux the torch wheel on PyPI is already the CUDA build -- it pulls
`cuda-toolkit`, `cudnn` and `nccl` with it -- so the GPU works out of the
box and no extra index is needed. You only need
https://pytorch.org for a *different* CUDA version, or for the CPU-only
build, which is the smaller download; install either first and the line
above will leave it alone.

Developed against Python 3.14 and torch 2.11; 3.12 is the safer default if
you have no reason to prefer another.

## Running the world action model

A trained multi-agent world action model is in the repository, so this
works from a fresh clone with nothing else collected or trained:

```bash
# watch it play itself, and write a gif
python examples/play/rollout_wam.py \
    --model marlenv/demodata/wam_deep/model_step8000.pt \
    --checkpoint marlenv/demodata/az_policy.pt \
    --out showcase/selfplay.gif

# play it yourself: arrows or WASD steer, the others are the model's
python examples/play/play_wam.py \
    --model marlenv/demodata/wam_deep/model_step8000.pt \
    --checkpoint marlenv/demodata/az_policy.pt
```

Both bootstrap with real steps from the simulator before handing over, so
the model starts from something in distribution. `--checkpoint` is the
search that plays that prefix, and it matters more than it looks: without
it the prefix is played by random rollouts, which survive but rarely take
fruit, so the model is handed a short-snaked history unlike anything it
trained on. `--bootstrap 1` hands over immediately, which is the hardest
case rather than the neutral one. `play_wam.py` needs a display -- add
`--headless` to run it without one.

`--background-gradient` must match the data the model was trained on, and
defaults to 16.0; pass `0` for a model trained on the gradient-free sets.
Nothing records this in the checkpoint, so it is on you to get right.

The model was trained with three snakes. It generalises to more, because
agents are told apart by where they are rather than by an identity
embedding, but the palette assigns a colour per snake and it has only ever
seen three of them. Cycle the colours instead of introducing new ones:

```bash
python examples/play/rollout_wam.py \
    --model marlenv/demodata/wam_deep/model_step8000.pt \
    --num-agents 6 --snake-colors 3
```

## Examples

The scripts are grouped by what they are for:

| folder | what is in it |
| --- | --- |
| `examples/data/` | collecting episodes, and repairing collected ones |
| `examples/train/` | the search policy, and the world models |
| `examples/play/` | playing, and recording rollouts to gif |
| `examples/analysis/` | scoring predictions, and palette diagnostics |

## Training

The datasets are not in the repository -- a seed and the collection script
regenerate them, and they are larger than the models. Collect first:

```bash
python examples/data/collect_dataset.py --preset expert  --episodes 1200 \
    --workers 20 --checkpoint marlenv/demodata/az_policy.pt
python examples/data/collect_dataset.py --preset explore --episodes 1500 \
    --workers 20 --checkpoint marlenv/demodata/az_policy.pt
```

Then train. Frames come from every component; actions only from the expert
one, because an exploration episode's actions are partly noise and the
policy head cannot fit noise:

```bash
python examples/train/train_wm_multi.py --context 48 --steps 36000 \
    --depth 12 --action-weight 0.075 0.0 --action-dropout 0.5 1.0 \
    --out marlenv/demodata/wam_new
```

`--action-weight` is per component and matters more than it looks: at 1.0
the action term takes eight times the frame term's gradient by the end of a
run, and it stops being able to spend it once the policy reaches its
entropy floor. The `pull a/f` figure in the log is that ratio, measured
rather than assumed.

### Growing a trained model deeper

Depth is what closed the gap to the single agent model, and it does not
need a retrain. `--init` with a deeper `--depth` grafts: each trained block
is followed by a silenced copy of itself, so step zero computes exactly
what the shallower model did and carries on from there. The target depth
must be a whole multiple of the checkpoint's.

```bash
# 12 blocks -> 24, carrying on from the trained model
python examples/train/train_wm_multi.py --init \
    marlenv/demodata/wam_deep/model_step8000.pt \
    --depth 24 --steps 8000 --lr 1e-4 \
    --action-weight 0.025 0.0 --action-dropout 0.5 1.0 \
    --out marlenv/demodata/wam_24
```

The context window is inherited from the checkpoint, so it does not have to
be repeated. Expect the loss to sit *above* where the shallower model
finished for the first couple of thousand steps: the duplicates start inert
and their inner weights only begin learning as their gates open. Going 12
to 24 doubles the parameters to 19.2M and roughly doubles the step time.

`--save-at-start` writes the graft before training touches it, which is how
to check that it really does reproduce what it grew from.

### Measuring it

Frame accuracy is not enough on its own. Most of a view is background and
wall, so a model can be right about nearly every pixel while its snakes
quietly dissolve, and the loss will say it is improving the whole time.

```bash
python examples/analysis/grade_ratchet.py \
    --models marlenv/demodata/wam_deep/model_step8000.pt \
             marlenv/demodata/flex_wam/model_step4000.pt \
    --names global flex \
    --policy marlenv/demodata/az_policy.pt
```

This rolls each model forward under the simulator's own actions and reports
two rates separately, one step at a time:

| | |
| --- | --- |
| `lost` | cells that hold a snake and were not drawn as one |
| `gained` | cells drawn as a snake that hold none |

The asymmetry is the measurement. A model whose errors run both ways sits
near the right length; one that only ever loses collapses in a rollout
however small its per-step error, because the shortened snake becomes its
own history and nothing ever puts a cell back. That is the difference
between a model that can be played and one that cannot, and the totals hide
it completely.

`dreamt` and `true` are the snake length visible from the agent's own
viewpoint, so they are a floor on the real length -- a long snake reaches
past the edge of the view.

#### Settings that will quietly ruin it

**`--policy` must be the one the data was collected with.** Without it the
simulator is driven by random rollouts, which survive but rarely take
fruit. The snakes then never grow past about three, no model is ever asked
to hold a long one, and the measurement finds nothing because nothing it
looks for happens. This is the single easiest way to get a confidently
wrong answer.

**`--background-gradient` must match the training data.** A model trained
without the gradient reads one as an observation it has never seen, and a
model trained with it reads a flat background the same way. It defaults to
16.0, which suits the original datasets and is wrong for anything collected
with `--background-gradient 0`. Nothing records this in the checkpoint or
the dataset, so it has to be passed by hand -- the same applies to
`--observation-noise` and `--snake-noise`.

**`--bootstrap` decides what history the model starts from.** One frame is
the hardest case and mostly measures a cold start; a prefix longer than
`--window` has its earliest frames evicted before the rollout begins, so
anything past the window is wasted. Somewhere well inside the window, with
the real policy driving it, is what a played rollout looks like.

**`--window` is not free to choose.** Longer is not better: measured on the
global model, own length at rollout steps 40-59 held 1.80 at a window of 12
and collapsed to 0.18 at 48, because a longer context keeps more
self-generated, already-shortened history in view to reinforce itself.

### Scoring single frames

```bash
python examples/analysis/grade_frames.py \
    --models marlenv/demodata/wm_ctx48/model.pt \
             marlenv/demodata/wam_deep/model_step8000.pt \
    --names single multi
```

Scores next-frame prediction split by cell type. Do not read the aggregate
loss on its own: most of a view is background and wall, which are easy, so
a model that renders those perfectly and smears every snake still scores
well.

## Rules


### Snake Game

Multiple snakes battle on a fixed size grid map.

Each snake is spawned at a random location on the map, with a random pose and direction at reset().

The map may be initialized with a different walls upon instantiation of the environment.

Snake dies when its head hits a wall or body of another snake. Here, the other snake receives a reward for kill and the dead snake receives a reward for death ('lose').

When multiple snakes collide head to head, all dies without receiving the kill score. 

When there is only one snake remaining, it receives a win reward for every unit time of survival.

The snake grows by one pixel when it has eatten a fruit. 

**Observation Types**

Image grid : The order is  **'NHWC'**

## Examples Input Arguments

### Snake Game

Creating an environment

```python
import gym
import marlenv
env = gym.make(
    'Snake-v1',
    height=20,       # Height of the grid map
    width=20,        # Width of the grid map
    num_snakes=4,    # Number of snakes to spawn on grid
    snake_length=3,  # Initial length of the snake at spawn time
    vision_range=5,  # Vision range (both width height), map returned if None
    frame_stack=1,   # Number of observations to stack on return
)
```

Single-agent wrapper

```python
env = gym.make('Snake-v1', num_snakes=1)
env = marlenv.wrappers.SingleAgent(env)
```

This will unwrap the returned the observation, reward, etc from a list

Using the make_snake() function

```python
# Automatically chooses wrappers to handle single agent, multi-agent, vector_env, etc.
env, observation_space, action_space, properties = marlenv.wrappers.make_snake(
    num_envs=1,  # Number of environments. Used to decided vector env or not
    num_snakes=1,  # Number of players. Used to determine single/multi agent
    **kwargs  # Other input parameters to the environment
)
```

The returned values are

- env : The environment object
- observation_space : The processed observation space (according to env type)
- action_space : The processed action space
- properties : The properties is a dict that includes
    - high: highest value that observation can have
    - low: lowest value that the observation can have
    - num_envs: number of environments
    - num_snakes: number of snakes to be spawned
    - discrete: True if action space is discrete, categorical
    - action_info
        - {action_high, action_low} if continuous action or {action_n} if discrete

**Custom reward function**

The user can change the reward function structure of the snake-game upon instantiation. 

The reward function can be defined using python dictionary as the following

```python
custom_reward_func = {
    'fruit': 1.0,
    'kill': 0.0,
    'lose': 0.0,
    'time': 0.0,
    'win': 0.0
}
env = gym.make('snake-v1', reward_func=custom_reward_func)
```

Each of the each of the keys represent

- fruit : reward received when the snake eats a fruit
- kill : reward received when the snake kills another snake
- lose : reward (or penalty) received when the snake dies
- time : reward received for each unit of time of survival
- win : reward received during the snake's time of survival as the last one standing

Each reward can be both + and - float number

## Testing

```python
pytest
```

## Citation

```python
@MISC{marlenv2021,
author =   {ML2},
title =    {Marlenv, Multi-agent Reinforcement Learning Environment},
howpublished = {\url{http://github.com/kc-ml2/marlenv}},
year = {2021}
}
```

## Updates

Currently, there is only one environment of multi-agent snake game.
