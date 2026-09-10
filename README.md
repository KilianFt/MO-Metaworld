# Meta-World — Adapted for Multi-Objective Reinforcement Learning (MORL)

## MORL extension (added in this fork)

This fork adds an effort objective to Meta-World's multi-task benchmarks. Install
this checkout with `uv sync` or `pip install -e .` to use the extension.

With `multi_objective=True`, both objectives are maximized:

$$r_t = [r_{\mathrm{task},t},\; -c\|a_t\|_2^2].$$

The task reward is unchanged. Effort follows
[MO-Gymnasium's squared-action convention](https://github.com/Farama-Foundation/MO-Gymnasium/blob/main/mo_gymnasium/envs/mujoco/half_cheetah_v5.py),
using all four commands (XYZ and gripper). It measures commanded effort, not
physical energy. `effort_weight=c` defaults to `1.0` and must be finite and positive.

```python
import metaworld

envs = metaworld.make_mt_envs(
    "MT10",  # also MT25 or MT50
    multi_objective=True,
    effort_weight=1.0,
    use_one_hot=True,
    seed=42,
)
try:
    obs, infos = envs.reset(seed=42)
    obs, rewards, terminated, truncated, infos = envs.step(
        envs.action_space.sample()
    )  # rewards: (10, 2), or (50, 2) for MT50
finally:
    envs.close()
```

MORL currently supports synchronous vectorization only. Episode returns retain
both objectives; with default autoreset, terminal statistics appear under
`infos["final_info"]["episode"]["r"]`. Scalar reward normalization and
`recurrent_info_in_obs` are unsupported in MORL mode. For fair comparisons, fix
the horizon, seeds, effort weight, and preference set, and report results per task.
Calls without `multi_objective=True` retain the original scalar rewards.

---

**Original Meta-World README**

[![Python](https://img.shields.io/pypi/pyversions/metaworld.svg)](https://badge.fury.io/py/metaworld)
[![PyPI](https://badge.fury.io/py/metaworld.svg)](https://badge.fury.io/py/metaworld.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2505.11289-b31b1b.svg)](https://arxiv.org/pdf/2505.11289)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://metaworld.farama.org/" target = "_blank">
    <img src="https://github.com/Farama-Foundation/Metaworld/blob/main/metaworld-text.png" width="500px"/>
</p>

Meta-World is an open source benchmark for developing and evaluating multi-task and meta reinforcement learning algorithms for continuous control robotic manipulation environments, with various benchmarks to evaluate different aspects of reinforcement learning algorithms.

The documentation website is at [metaworld.farama.org](https://metaworld.farama.org), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6

## Installation

To install Meta-World, use `pip install metaworld`

We support and test for Python 3.10, 3.11, 3.12, and 3.13 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

## API

The Meta-World API follows the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API for environment creation and environment interactions.

To create a benchmark and interact with it:

```python
import gymnasium as gym
import metaworld


env = gym.make("Meta-World/MT1", env_name="reach-v3")

observation, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Available Benchmarks

### Multi-Task Benchmarks
The MT1, MT10, and MT50 benchmarks are the Multi-Task Benchmarks. These benchmarks are used to learn a multi-task policy that can learn 1, 10, or 50 training tasks simultaneously. MT1 benchmarks can be created with any of the 50 tasks available in Meta-World.
In the MT10 and MT50 benchmarks, the observations returned by the benchmark will come with one-hot task IDs appended to the state.

### Meta-Learning Benchmarks
The ML1, ML10, and ML45 benchmarks are 3 meta-reinforcement learning benchmarks available in Meta-World. The ML1 benchmark can be used with any of the 50 tasks available in Meta-World.
The ML1 benchmark tests for few-shot adaptation to goal variations within a single task. The ML10 and ML45 both test few-shot adaptation to new tasks. ML10 comprises 10 train tasks with 5 test tasks, while ML45 comprises of 45 training tasks with 5 test tasks.


## Creating Multi-Task Benchmarks

### MT1
```python
import gymnasium as gym
import metaworld

seed = 42 # for reproducibility

env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed) # MT1 with the reach environment

obs, info = env.reset()

a = env.action_space.sample() # randomly sample an action
obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action
```

### MT10
MT10 has two different versions that can be returned by ```gym.make```. The first version is the synchronous version of the benchmark where all environments are contained within the same process.
For users with limited compute resources, the synchronous option needs the least resources.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make_vec('Meta-World/MT10', vector_strategy='sync', seed=seed) # this returns a Synchronous Vector Environment with 10 environments

obs, info = envs.reset() # reset all 10 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 10 environments
```
Alternatively, for users with more compute we also provide the asynchronous version of the MT10 benchmark where each environment is isolated in it's own process and must use inter-process messaging via pipes to communicate.

```python
envs = gym.make_vec('Meta-World/MT10', vector_strategy='async', seed=seed) # this returns an Asynchronous Vector Environment with 10 environments
```

### MT50
MT50 also contains two different versions, a synchronous and an asynchronous version, of the environments.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make_vec('Meta-World/MT50', vector_strategy='sync', seed=seed) # this returns a Synchronous Vector Environment with 50 environments

obs, info = envs.reset() # reset all 50 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 50 environments
```

```python
envs = gym.make_vec('Meta-World/MT50', vector_strategy='async', seed=seed) # this returns an Asynchronous Vector Environment with 50 environments
```


## Meta-Learning Benchmarks
Each Meta-reinforcement learning benchmark has training and testing environments. These environments must be created separately as follows.

### ML1
```python
import gymnasium as gym
import metaworld

seed = 42


train_envs = gym.make('Meta-World/ML1-train', env_name='reach-V3', seed=seed)
test_envs = gym.make('Meta-World/ML1-test', env_name='reach-V3', seed=seed)

# training procedure use train_envs
# testing procedure use test_envs

```


### ML10
Similar to the Multi-Task benchmarks, the ML10 and ML45 environments can be run in synchronous or asynchronous modes.


```python
import gymnasium as gym
import metaworld

train_envs = gym.make_vec('Meta-World/ML10-train', vector_strategy='sync', seed=seed) # or vector_strategy='async'
test_envs = gym.make_vec('Meta-World/ML10-test', vector_strategy='sync', seed=seed) # or vector_strategy='async'
```


### ML45
```python
import gymnasium as gym
import metaworld

train_envs = gym.make_vec('Meta-World/ML45-train', vector_strategy='sync', seed=seed) # or vector_strategy='async'
test_envs = gym.make_vec('Meta-World/ML45-test', vector_strategy='sync', seed=seed) # or vector_strategy='async'
```


## Custom Benchmarks
Finally, we also provide support for creating custom benchmarks by combining any number of Meta-World environments.

The prefix 'mt' will return environments that are goal observable for Multi-Task reinforcement learning, while the prefix 'ml' will return environments that are partially observable for Meta-reinforcement learning.
Like the included MT and ML benchmarks, these environments can also be run in synchronous or asynchronous mode.
In order to create a custom benchmark, the user must provide a list of environment names with the suffix '-v3'.

```python
import gymnasium as gym
import metaworld

envs = gym.make_vec('Meta-World/custom-mt-envs',vector_strategy='sync', envs_list=['env_name_1-v3', 'env_name_2-v3', 'env_name_3-v3'], seed=seed) # or vector_strategy='async'
envs = gym.make_vec('Meta-World/custom-ml-envs',vector_strategy='sync', envs_list=['env_name_1-v3', 'env_name_2-v3', 'env_name_3-v3'], seed=seed) # or vector_strategy='async'
```

## Development Roadmap

We have a roadmap for future development work for Gymnasium available here: https://github.com/Farama-Foundation/Metaworld/issues/500

## Benchmark Code

Code for producing results found in the paper can be found at: https://github.com/rainx0r/metaworld-algorithms

## Citation

You can cite Meta-World using our related paper (https://openreview.net/forum?id=1de3azE606) as:

```
@inproceedings{
mclean2025metaworld,
title={Meta-World+: An Improved, Standardized, {RL} Benchmark},
author={Reginald McLean and Evangelos Chatzaroulas and Luc McCutcheon and Frank R{\"o}der and Tianhe Yu and Zhanpeng He and K.R. Zentner and Ryan Julian and J K Terry and Isaac Woungang and Nariman Farsad and Pablo Samuel Castro},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=1de3azE606}
}
```
