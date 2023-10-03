# import the environment wrapper and gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import DeterministicMixin, GaussianOptionsMixin, Model
from skrl.trainers.torch import ManualTrainer
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
import gymnasium as gym
from skrl.agents.torch.soc import SOC, SOC_DEFAULT_CONFIG

# load the environment
env = gym.make('Pendulum-v1', render_mode="rgb_array")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'

## Define agent
class Actor(GaussianOptionsMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", num_options=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianOptionsMixin.__init__(self, num_options, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.option_action_layer = nn.Linear(300, self.num_options*self.num_actions)
        #TODO sett inn options-størrelse for layer
        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions*self.num_options))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        x = self.option_action_layer(x)

        # Reduce dim based on selected options
        x = x.reshape(-1, self.num_actions, self.num_options)
        options = inputs["options"].reshape(-1, self.num_actions, 1)
        x = x.gather(-1, options).squeeze(-1)
        # log_std_parameter = self.log_std_parameter.unsqueeze(-1).gather(-1, inputs["options"])

        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(x), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}
    

# Intra-Option-value network
class IntraOptionCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, N_options=2, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, N_options)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        q = F.relu(self.linear_layer_3(x))
        # return self.linear_layer_3(x), {}
        return q.gather(-1, inputs["taken_options"]), {}
    

# Option-value network
class OptionCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, N_options=2, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, N_options)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}

# instantiate a memory as experience replay
device = env.device
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)

# import the agent and its default configuration
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
models["critic_Qw"] = OptionCritic(env.observation_space, env.action_space, device, N_options=2,)  # only required during training
models["critic_Qu"] = IntraOptionCritic(env.observation_space, env.action_space, device, N_options=2,)  # only required during training
models["target_critic_Qw"] = OptionCritic(env.observation_space, env.action_space, device, N_options=2)  # only required during training
models["target_critic_Qu"] = IntraOptionCritic(env.observation_space, env.action_space, device, N_options=2)  # only required during training

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/soc.html#configuration-and-hyperparameters
cfg = SOC_DEFAULT_CONFIG.copy()
cfg["timesteps"] = 10
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0# cfg["learning_starts"] = 1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75
cfg["experiment"]["checkpoint_interval"] = 750
cfg["experiment"]["directory"] = "runs/torch/Pendulum"

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = SOC(models=models,
            memory=memory,  # only required during training
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

trainer = ManualTrainer(env=env, agents=agent, cfg=cfg)

# Run agent
# train the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.train(timestep=timestep)

# evaluate the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.eval(timestep=timestep)