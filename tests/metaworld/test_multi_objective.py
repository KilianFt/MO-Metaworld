"""Smoke coverage for multi-task, multi-objective rollouts."""

import gymnasium as gym
import numpy as np
import pytest

import metaworld
from metaworld.multi_objective import MOSyncVectorEnv, MultiObjectiveReward


@pytest.fixture(autouse=True)
def few_goals(monkeypatch):
    # Exercise the real task generator without generating 2,500 goals per test.
    monkeypatch.setattr(metaworld, "_N_GOALS", 2)


@pytest.mark.parametrize("benchmark,n", [("MT10", 10), ("MT50", 50)])
@pytest.mark.parametrize("mode", list(gym.vector.AutoresetMode))
def test_multi_task_rollout(benchmark, n, mode):
    env = gym.make_vec(
        f"Meta-World/{benchmark}",
        seed=0,
        multi_objective=True,
        use_one_hot=True,
        max_episode_steps=2,
        autoreset_mode=mode,
    )
    try:
        assert isinstance(env, MOSyncVectorEnv)
        obs, _ = env.reset(seed=0)
        np.testing.assert_array_equal(obs[:, -n:], np.eye(n))
        actions = np.full((n, 4), 0.5, dtype=np.float32)
        returns = np.zeros((n, 2))
        for _ in range(2):
            _, reward, term, trunc, info = env.step(actions)
            assert reward.shape == (n, 2)
            assert reward.dtype == np.float32
            assert np.isfinite(reward).all()
            np.testing.assert_array_equal(reward[:, 1], -np.ones(n))
            returns += reward
        assert trunc.all()
        assert not term.any()
        final_info = (
            info["final_info"] if mode == gym.vector.AutoresetMode.SAME_STEP else info
        )
        np.testing.assert_allclose(final_info["episode"]["r"], returns)
        assert "success" in final_info
        if mode == gym.vector.AutoresetMode.DISABLED:
            env.reset()
        _, reward, _, _, _ = env.step(actions)
        assert reward.shape == (n, 2)
        if mode == gym.vector.AutoresetMode.NEXT_STEP:
            np.testing.assert_array_equal(reward, np.zeros((n, 2)))
    finally:
        env.close()


def test_task_reward_and_dynamics_unchanged():
    kwargs = dict(name="reach-v3", seed=3, max_episode_steps=3)
    scalar = metaworld.make_mt_envs(**kwargs)
    mo = metaworld.make_mt_envs(**kwargs, multi_objective=True, effort_weight=0.25)
    try:
        np.testing.assert_array_equal(scalar.reset()[0], mo.reset()[0])
        for action in [
            np.zeros(4),
            np.array([0.2, -0.3, 0.4, 1.0], dtype=np.float32),
            np.ones(4),
        ]:
            obs, reward, term, trunc, info = scalar.step(action)
            mo_obs, mo_reward, mo_term, mo_trunc, mo_info = mo.step(action)
            np.testing.assert_array_equal(obs, mo_obs)
            np.testing.assert_allclose(
                mo_reward, [reward, -0.25 * np.square(action).sum()]
            )
            assert (term, trunc) == (mo_term, mo_trunc)
            assert info["success"] == mo_info["success"]
            assert mo.unwrapped.reward_space.contains(mo_reward)
    finally:
        scalar.close()
        mo.close()


@pytest.mark.parametrize("action", [np.ones(3), np.full(4, np.nan), np.full(4, 1.01)])
def test_invalid_actions(action):
    env = metaworld.make_mt_envs("reach-v3", multi_objective=True)
    try:
        env.reset()
        with pytest.raises(ValueError, match="finite action"):
            env.step(action)
        assert env.unwrapped.curr_path_length == 0
    finally:
        env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vector_strategy": "async"},
        {"effort_weight": -1},
        {"effort_weight": float("nan")},
        {"reward_normalization_method": "gymnasium"},
        {"recurrent_info_in_obs": True},
    ],
)
def test_unsupported_options(kwargs):
    with pytest.raises(ValueError):
        metaworld.make_mt_envs("reach-v3", multi_objective=True, **kwargs)


def test_class_based_api():
    benchmark = metaworld.MT1("reach-v3", seed=0)
    env = MultiObjectiveReward(benchmark.train_classes["reach-v3"]())
    try:
        env.set_task(benchmark.train_tasks[0])
        env.reset()
        assert env.step(np.ones(4))[1][1] == -4
    finally:
        env.close()


def test_custom_mt_and_scalar_vector():
    mo = gym.make_vec(
        "Meta-World/custom-mt-envs",
        vector_strategy="sync",
        envs_list=["reach-v3", "push-v3"],
        seed=2,
        multi_objective=True,
        use_one_hot=True,
        max_episode_steps=1,
    )
    scalar = metaworld.make_mt_envs("MT10", seed=2, max_episode_steps=1)
    try:
        mo.reset()
        assert mo.step(np.zeros((2, 4)))[1].shape == (2, 2)
        scalar.reset()
        assert scalar.step(np.zeros((10, 4)))[1].shape == (10,)
    finally:
        mo.close()
        scalar.close()
