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

### Learning a multi-agent model from one agent's record

Three runs, same architecture and same episodes, differing only in what
each is allowed to see. The question is whether a model can learn multi
agent dynamics from a single agent's account of them, deducing the rest.

```bash
# ceiling: every agent's record, nothing deduced
python examples/train/train_flex_wam.py --components expert_nogradient explore_nogradient ...

# the experiment: one agent's record, the others recovered while in view
python examples/train/train_flex_wam.py --egocentric ...

# baseline: one agent's record and nothing else
python examples/train/train_flex_wam.py --solo ...
```

None of the three is fully observed. Every agent sees the same 9x9 window
in all of them; what changes is how many records reach the model and how
much of the rest it has to work out. The egocentric run keeps other agents
only while their heads are in view, gives each visit a fresh identity, and
marks what it could not see with noise rather than dropping it.

The baseline is what makes the result readable. This architecture hands the
model the geometry: positions are axial RoPE over time, row and column, so
two agents' tokens carry their true spatial offset without anyone having to
learn it. A model might therefore place agents consistently because the
embedding says where they are, not because the data taught it. The solo run
is never shown a second agent, so whatever multi-agent consistency it still
produces is the geometry's doing and not the data's.

Read it with a metric that measures agreement *between* agents rather than
coherence of the stitched canvas: the canvas is assembled from dead
reckoned poses, so it looks coherent for a solo model too. What separates
them is whether two agents' generated views agree where they cover the same
world cells at the same step -- the same overlap check
`diagram_gen`/`examples/analysis` use on the real data, run on generated
frames instead.

A stronger version of this experiment would drop the geometric embedding
for a learned one and make the model find the spatial relationship in the
data. That is a separate run and is not set up here.

#### Inter-agent agreement: how it is measured

Every other measure in this repository scores one agent's view against the
truth. This one scores two agents against *each other*, and it exists
because the architecture could be answering the question on the model's
behalf. Positions are axial RoPE over time, row and column, so two agents'
tokens already carry their true spatial offset; a model could place a snake
in the same spot in both views because the embedding says where it is,
never having learned that two accounts of one board must agree.

`marlenv/grading/consistency.py`, driven by
`examples/analysis/grade_consistency.py`:

1. Roll the model forward from a bootstrap prefix played by the search that
   collected the data, on a board with the same settings the data was
   collected under. Every agent's action at every step is **forced to the
   simulator's**, so dream and truth stay on one trajectory.
2. At each step take every agent's dreamt view. These are north-up already,
   the frame the model works in, so no rotation is applied -- passing a
   north pose to `unrotate_view` is the identity, checked rather than
   assumed.
3. For each unordered pair of *living* agents, intersect the two views'
   world footprints. Footprints come from dead reckoning, which is exact
   here precisely because the actions were forced: the overlap is where the
   views really meet, not where a drifting model imagines they do. Pairs
   whose views do not meet contribute nothing; that is absence of evidence,
   not disagreement.
4. Decode both windows to palette classes and add the pair to a confusion
   matrix of (class one agent drew, class the other drew). The diagonal is
   agreement.

Two properties make the number readable.

**The truth is scored the same way, alongside.** Real views of one board
agree with each other perfectly, so this control must return 1.0000. It
does, for every arm. A harness that lined views up wrongly would fail there
first, rather than being read as a model that contradicts itself -- which
is the mistake this measurement is most likely to produce.

**The headline is agreement over snake cells only.** Background and wall
are roughly five sixths of any overlap, so the total is mostly a count of
empty board, and a model that draws less snake scores better on it for that
reason and no other. Snake cells are where the question lives.

Sample sizes below: 12 episodes, 60 steps, giving 1552 agent pairs and
29218 overlapping cells; identical for every arm, since the environment,
the bootstrap policy and the per-episode generators are all seeded and the
harness is deterministic. Rows measured in separate invocations are
therefore directly comparable, which is why each arm was measured once
rather than re-measured together.

#### What the arms scored

Agreement over snake cells; ratchet columns at rollout steps 45-59, where
the true length is 7.90; survival is how many of 240 rollout steps pass
before every viewpoint is retired, at `--death-patience 1`.

Each arm was trained 24000 steps and then warm started for a further 16000
at a third the learning rate. What that warm start changed is **not the
same for every arm**: the two single-record arms gained the observer's
death frames along with the extra steps, while the ceiling gained only the
steps, because the rectangular path selects on alive-or-trained and has
always kept the aftermath frame. The trio is matched on budget, not on
treatment.

| arm | steps | agreement | dreamt length | lost | gained | survival |
| --- | --- | --- | --- | --- | --- | --- |
| solo | 24000 | 0.016 | 7.86 | 0.413 | 0.114 | - |
| egocentric | 24000 | 0.800 | 11.64 | 0.119 | 0.510 | - |
| ceiling | 24000 | 0.852 | 5.93 | 0.321 | 0.025 | 28 |
| solo | 40000 | 0.020 | 5.07 | 0.613 | 0.064 | 181 |
| egocentric | 40000 | 0.748 | 7.62 | 0.235 | 0.269 | 240 |
| ceiling | 40000 | 0.876 | 4.65 | 0.463 | 0.062 | 204 |

The agreement column answers the confound and keeps answering it. The solo
model carries the same embedding and is handed the same true offset between
any two agents, and it agrees with itself about snake cells 1.6% of the
time -- one view draws a snake where the other draws bare board. Sixteen
thousand further steps move that to 2.0%. The geometry does not produce
inter-agent consistency at any training budget, so the 0.75 to 0.80 above
it was learned from data, and from one agent's record at that.

At matched budget the egocentric model reaches 85% of the ceiling's
agreement, which is less flattering than the 94% the two 24000-step
checkpoints happened to show. Both pairings are in the table because
picking the better one would be picking a number.

No arm is best at everything, and the disagreements between columns are the
point. The egocentric model is the only one whose dreamt length tracks the
truth and the only one to survive all 240 steps, while the ceiling is the
most self-consistent and dreams snakes barely half the true length. The
solo model at 24000 steps tracked length better than either while agreeing
with itself almost never. **Length is a marginal statistic and agreement is
a joint one; a model can get the first right with no coherent joint
structure whatever.** Any single column here tells a false story.

Canvas coverage tells no story at all: 0.87, 0.86 and 0.84 for the three
24000-step arms. It cannot separate them, and it ranks the solo model
first, because the canvas is stitched from dead reckoned poses and so looks
coherent whatever the model believes about other agents. That is the
measurement inter-agent agreement had to replace.

Survival separates the arms only at `--death-patience 1`. At 2 and 3 every
arm runs the full 240 steps, so a rollout at the default setting shows
three models that look equally healthy.

#### What the warm starts do and do not show

Both warm starts move the same way on the ratchet: more `lost`, less
`gained`. The egocentric model was over-drawing, so the shift put it on the
true length; the solo model was balanced, so the same shift pushed it into
heavy erosion. A death frame is a target in which a snake stops existing,
which is gradient toward drawing less snake, and that is consistent with a
uniform shift whose sign of benefit depends only on where a model started.
Restoring the death frames is a bias shift, not a repair.

Neither warm start separates the death frames from 16000 further steps at a
lower learning rate. A continuation with `--drop-death-frames` would be the
control and has not been run. The wall agreement fell in both arms (0.979
to 0.871, and 0.898 to 0.853), which at least says that regression is not
something about egocentric data.

The ceiling's own warm start is the closest thing to that control that
exists here, since for it the death frames were never the variable. It
gained 16000 steps and nothing else, and it moved a long way: agreement
from 0.852 to 0.876, survival from 28 steps to 204, and erosion deeper,
dreaming 4.65 against a true 7.90 where it had dreamt 5.93. Extra steps
alone are clearly not inert, so the egocentric arm's improvement cannot be
attributed to the death frames on the strength of these runs.

The ceiling also trained on the fatal actions the collector was recording
wrongly at the time, in both of its runs. Comparisons against it that turn
on how an agent dies are not fair and are not made here.

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
