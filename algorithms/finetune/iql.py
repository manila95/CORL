# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import minari
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cpu"
    env: str = "mujoco/hopper/medium-v0"   # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(5e4)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    actor_dropout: float = 0.0  # Dropout in actor network
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    expl_noise: float = 0.03  # Std of Gaussian exploration noise
    noise_clip: float = 0.5  # Range to clip noise
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    # Wandb logging
    project: str = "CORL"
    group: str = "IQL-Minari"
    name: str = "IQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    # Gymnasium requires observation_space argument for TransformObservation
    env = gym.wrappers.TransformObservation(env, normalize_state, env.observation_space)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
        # raise NotImplementedError


def minari_to_qlearning_dataset(minari_dataset) -> Dict[str, np.ndarray]:
    """Convert Minari dataset to qlearning format (D4RL-like format).
    
    Args:
        minari_dataset: Minari dataset object
        
    Returns:
        Dictionary with keys: observations, actions, rewards, next_observations, terminals
    """
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    # Iterate through all episodes in the dataset
    for episode_id in minari_dataset.episode_indices:
        episode = minari_dataset[episode_id]
        
        # Minari episode data structure
        # Access observations, actions, rewards, terminations, truncations
        episode_obs = np.array(episode.observations, dtype=np.float32)
        episode_actions = np.array(episode.actions, dtype=np.float32)
        episode_rewards = np.array(episode.rewards, dtype=np.float32)
        
        # Handle terminations and truncations
        if hasattr(episode, 'terminations'):
            episode_terminations = np.array(episode.terminations, dtype=bool)
        else:
            episode_terminations = np.zeros(len(episode_obs), dtype=bool)
        
        if hasattr(episode, 'truncations'):
            episode_truncations = np.array(episode.truncations, dtype=bool)
        else:
            episode_truncations = np.zeros(len(episode_obs), dtype=bool)
        
        # Create terminals: True if episode ends (termination or truncation)
        episode_dones = episode_terminations | episode_truncations
        
        # For each step in the episode, create a transition
        # We create transitions for all steps except the last one
        for i in range(len(episode_obs) - 1):
            observations.append(episode_obs[i])
            actions.append(episode_actions[i])
            rewards.append(episode_rewards[i])
            next_observations.append(episode_obs[i + 1])
            terminals.append(episode_dones[i])
    
    return {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.float32),
        "terminals": np.array(terminals, dtype=np.float32),
    }


def set_env_seed(env: Optional[gym.Env], seed: int):
    # Gymnasium uses reset(seed=seed) instead of seed()
    env.reset(seed=seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    # wandb.run.save() is no longer needed in newer wandb versions
    # The run is automatically saved when wandb.init() is called


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    # Gymnasium uses reset(seed=seed) instead of seed()
    env.reset(seed=seed)
    actor.eval()
    episode_rewards = []
    successes = []
    terminations = []
    for _ in range(n_episodes):
        # Gymnasium reset() returns (observation, info) tuple
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        goal_achieved = False
        episode_terminated = False
        while not done:
            action = actor.act(state, device)
            # Gymnasium step() returns (observation, reward, terminated, truncated, info)
            state, reward, terminated, truncated, env_infos = env.step(action)
            done = terminated or truncated
            if terminated:
                episode_terminated = True
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)
        terminations.append(float(episode_terminated))

    actor.train()
    termination_count = int(sum(terminations))
    termination_rate = np.mean(terminations)
    return np.asarray(episode_rewards), np.mean(successes), termination_count, termination_rate


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    # Load Minari dataset
    # Minari uses the same naming convention for D4RL datasets
    minari_dataset = minari.load_dataset(config.env, download=True)
    
    # Convert Minari dataset to qlearning format
    dataset = minari_to_qlearning_dataset(minari_dataset)
    
    # Create environment for evaluation
    # Extract base environment name (e.g., "halfcheetah-medium-expert-v2" -> "halfcheetah")
    # For most MuJoCo envs, we can use the dataset's env_spec
    try:
        # Try to get environment from Minari dataset metadata
        env_spec = minari_dataset.env_spec
        if env_spec is not None:
            env = gym.make(env_spec.id)
            eval_env = gym.make(env_spec.id)
        else:
            # Fallback: extract base name from config.env
            base_name = config.env.split("-")[0]
            env = gym.make(f"{base_name}-v0")
            eval_env = gym.make(f"{base_name}-v0")
    except Exception:
        # Final fallback: try to create env from config name directly
        # Remove dataset suffix if present (e.g., "-medium-expert-v2" -> "")
        base_name = config.env.split("-medium")[0].split("-expert")[0].split("-random")[0].split("-replay")[0]
        env = gym.make(f"{base_name}-v0")
        eval_env = gym.make(f"{base_name}-v0")

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.offline_iterations,
    }

    print("=" * 70)
    print(f"Training IQL (Finetune)")
    print("=" * 70)
    print(f"Environment: {config.env}")
    print(f"Seed: {seed}, Eval Seed: {config.eval_seed}")
    print(f"Device: {config.device}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Offline iterations: {config.offline_iterations:,}")
    print(f"Online iterations: {config.online_iterations:,}")
    print(f"Total iterations: {config.offline_iterations + config.online_iterations:,}")
    print(f"Batch size: {config.batch_size}")
    print(f"Evaluation frequency: {config.eval_freq:,} steps")
    print(f"Beta: {config.beta}, IQL tau: {config.iql_tau}")
    print("=" * 70)

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    
    # Gymnasium reset() returns (observation, info) tuple
    state, _ = env.reset()
    done = False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []
    train_terminations = []  # Track terminations during online training
    total_train_terminations = 0
    total_train_episodes = 0
    
    start_time = time.time()
    total_iterations = int(config.offline_iterations) + int(config.online_iterations)
    log_freq = max(100, config.eval_freq // 10)  # Log progress every 10% of eval_freq

    print("\nStarting training...")
    print("-" * 70)
    print("Offline pretraining phase")
    print("-" * 70)
    
    for t in range(total_iterations):
        if t == config.offline_iterations:
            print("\n" + "=" * 70)
            print("Switching to Online tuning phase")
            print("=" * 70)
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            if not config.iql_deterministic:
                action = action.sample()
            else:
                noise = (torch.randn_like(action) * config.expl_noise).clamp(
                    -config.noise_clip, config.noise_clip
                )
                action += noise
            action = torch.clamp(max_action * action, -max_action, max_action)
            action = action.cpu().data.numpy().flatten()
            # Gymnasium step() returns (observation, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, env_infos = env.step(action)
            done = terminated or truncated

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward

            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True
                # Track terminations during online training
                if terminated:
                    total_train_terminations += 1

            if config.normalize_reward:
                reward = modify_reward_online(reward, config.env, **reward_mod_dict)

            replay_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state
            if done:
                # Track episode completion
                total_train_episodes += 1
                # Gymnasium reset() returns (observation, info) tuple
                state, _ = env.reset()
                done = False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                # Minari doesn't have get_normalized_score, so we use raw score
                online_log["train/episode_return_normalized"] = episode_return
                online_log["train/episode_length"] = episode_step
                # Track termination rate
                if total_train_episodes > 0:
                    termination_rate = total_train_terminations / total_train_episodes
                    online_log["train/termination_rate"] = termination_rate
                    online_log["train/total_terminations"] = total_train_terminations
                    online_log["train/total_episodes"] = total_train_episodes
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        wandb.log(log_dict, step=trainer.total_it)
        
        # Progress logging
        if (t + 1) % log_freq == 0:
            elapsed_time = time.time() - start_time
            progress = (t + 1) / total_iterations * 100
            phase = "Online" if t >= config.offline_iterations else "Offline"
            steps_per_sec = (t + 1) / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (total_iterations - (t + 1)) / steps_per_sec if steps_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            
            print(f"\n[{t + 1:,}/{total_iterations:,}] ({progress:.1f}%) [{phase}] | "
                  f"Time: {elapsed_time/60:.1f}m | "
                  f"Speed: {steps_per_sec:.1f} steps/s | "
                  f"ETA: {eta_minutes:.1f}m")
            print(f"  Q Loss: {log_dict.get('q_loss', 0):.4f} | "
                  f"V Loss: {log_dict.get('v_loss', 0):.4f} | "
                  f"Actor Loss: {log_dict.get('actor_loss', 0):.4f}")
            if t >= config.offline_iterations:
                term_rate = online_log.get('train/termination_rate', 0)
                term_count = online_log.get('train/total_terminations', 0)
                term_episodes = online_log.get('train/total_episodes', 0)
                print(f"  Episode Return: {online_log.get('train/episode_return', 0):.3f} | "
                      f"Episode Length: {online_log.get('train/episode_length', 0):.0f}")
                if term_episodes > 0:
                    print(f"  Terminations: {term_count}/{term_episodes} ({term_rate:.1%})")
        
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            eval_start_time = time.time()
            eval_progress = (t + 1) / total_iterations * 100
            phase = "Online" if t >= config.offline_iterations else "Offline"
            print("\n" + "=" * 70)
            print(f"Evaluation at step {t + 1:,} ({eval_progress:.1f}% complete) [{phase}]")
            print("=" * 70)
            
            eval_scores, success_rate, eval_termination_count, eval_termination_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.eval_seed,
            )
            eval_score = eval_scores.mean()
            eval_std = eval_scores.std()
            eval_min = eval_scores.min()
            eval_max = eval_scores.max()
            
            eval_log = {}
            # Minari doesn't have get_normalized_score, so we use raw score
            normalized_eval_score = eval_score  # Use raw score
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(eval_successes))
                eval_log["eval/success_rate"] = success_rate
            eval_log["eval/score"] = normalized_eval_score
            eval_log["eval/termination_count"] = eval_termination_count
            eval_log["eval/termination_rate"] = eval_termination_rate
            evaluations.append(normalized_eval_score)
            
            eval_time = time.time() - eval_start_time
            
            print(f"Evaluation Results ({config.n_episodes} episodes, {eval_time:.1f}s):")
            print(f"  Mean Score: {eval_score:.3f} ± {eval_std:.3f}")
            print(f"  Min Score: {eval_min:.3f}, Max Score: {eval_max:.3f}")
            print(f"  Terminations: {eval_termination_count}/{config.n_episodes} ({eval_termination_rate:.1%})")
            if is_env_with_goal and t >= config.offline_iterations:
                print(f"  Success Rate: {success_rate:.3f}")
            if len(evaluations) > 1:
                print(f"  Best Score So Far: {max(evaluations):.3f}")
                print(f"  Improvement: {evaluations[-1] - evaluations[0]:.3f}")
            print("=" * 70)
            
            if config.checkpoints_path is not None:
                checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{t + 1}.pt")
                torch.save(trainer.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            wandb.log(eval_log, step=trainer.total_it)
            print()  # Empty line for readability
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Total iterations: {total_iterations:,}")
    print(f"Average speed: {total_iterations/total_time:.1f} steps/s")
    if evaluations:
        print(f"\nFinal Evaluation Score: {evaluations[-1]:.3f}")
        print(f"Best Evaluation Score: {max(evaluations):.3f}")
        print(f"Average Evaluation Score: {sum(evaluations)/len(evaluations):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    train()
