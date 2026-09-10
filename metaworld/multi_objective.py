"""Multi-objective task reward and commanded-control effort interfaces."""

import gymnasium as gym
import numpy as np


class MultiObjectiveReward(gym.Wrapper):
    """Return ``[task_reward, -effort_weight * sum(action**2)]``.

    All four normalized commands (XYZ displacement and gripper) contribute.
    This is a control-effort proxy, not mechanical work or joint torque.
    Actions must be finite and within the environment's action space; they are
    neither rescaled nor clipped. The original task reward and info are retained.
    """

    reward_dim = 2
    objective_names = ("task", "effort")

    def __init__(self, env: gym.Env, effort_weight: float = 1.0):
        super().__init__(env)
        if not np.isfinite(effort_weight) or effort_weight <= 0:
            raise ValueError("effort_weight must be finite and strictly positive")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("MultiObjectiveReward requires a Box action space")
        self.effort_weight = float(effort_weight)
        self.reward_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, 0.0], dtype=np.float32),
            dtype=np.float32,
        )
        # MO-Gymnasium wrappers discover the reward space on the unwrapped env.
        self.unwrapped.reward_space = self.reward_space
        self.unwrapped.reward_dim = self.reward_dim
        self.unwrapped.objective_names = self.objective_names

    def step(self, action):
        action = np.asarray(action)
        if (
            action.shape != self.action_space.shape
            or not np.all(np.isfinite(action))
            or np.any(action < self.action_space.low)
            or np.any(action > self.action_space.high)
        ):
            raise ValueError("Expected a finite action within the action space")
        effort = -self.effort_weight * float(
            np.sum(np.square(action.astype(np.float64)))
        )
        obs, task_reward, terminated, truncated, info = self.env.step(action)
        reward = np.array([task_reward, effort], dtype=np.float32)
        info = dict(info, task_reward=float(task_reward), effort_reward=effort)
        return obs, reward, terminated, truncated, info

    def set_task(self, task):
        """Preserve the class-based benchmark's explicit task selection API."""
        return self.unwrapped.set_task(task)


class MORecordEpisodeStatistics(gym.wrappers.RecordEpisodeStatistics):
    """Accumulate a separate episode return for each objective."""

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.episode_returns = np.zeros(2, dtype=np.float64)
        return result


class MOSyncVectorEnv(gym.vector.SyncVectorEnv):
    """Gymnasium synchronous vectorization with an (env, objective) reward buffer.

    Uses Gymnasium's reset/step logic, including all three autoreset modes.
    """

    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)
        self.reward_space = self.envs[0].get_wrapper_attr("reward_space")
        self.single_reward_space = self.reward_space
        self.reward_dim = 2
        self.objective_names = MultiObjectiveReward.objective_names
        self._rewards = np.zeros((self.num_envs, self.reward_dim), dtype=np.float32)


def mt_vectorizer(strategy, multi_objective=False):
    """Select a vectorizer before constructing any expensive environments."""
    if strategy not in ("sync", "async"):
        raise ValueError("vector_strategy must be 'sync' or 'async'")
    if multi_objective:
        if strategy != "sync":
            raise ValueError(
                "Multi-objective environments require vector_strategy='sync'"
            )
        return MOSyncVectorEnv
    return getattr(gym.vector, f"{strategy.capitalize()}VectorEnv")
