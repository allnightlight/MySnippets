#!/usr/bin/env python
# coding: utf-8

# * This document is written, referring to the following tutorials about tf_agents: 
#     * [tutorials](https://github.com/tensorflow/agents/tree/master/docs/tutorials)

# In[ ]:


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.networks import network
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


# # 1. Define an environment class as an inheritance from PyEnvironment
# Instances from this class represent first-order delay(FOD) systems.

# In[ ]:


class MyEnv(py_environment.PyEnvironment):
    '''
    
    Y(s) = K/(1+T*s) * U(s)
    
    T * dy(t)/dt = - y(t) + K * u(t), t > 0, 
    y(0) = y_init.
    
    y(t+1) = (1-1/T) * y(t) + K / T * u(t), t = 1,2, ...
    y(0) = y_init.
    
    x(t+1) = (1-1/T) * x(t) + K / T * u(t), t = 1,2, ...
    x(0) = x_init, 
    y(t) = x(t).
    
    '''

    def __init__(self, nStepSimulation = 100, T = 10, K = 1.0, discount = 0.9):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, minimum=(-1,), maximum=(1,), name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, minimum=(-1,), maximum=(1,), name='observation')

        self._state = self.getInitialState()
        self._episode_ended = False
        self.time = 0
        self.nStepSimulation = nStepSimulation
        self.T = T
        self.K = K
        self.discount = discount
    
    def getInitialState(self):
        return 0.
    
    def getObservation(self):
        return np.array((self._state,), np.float32) # (1,)
    
    def getReward(self):
        sv = 1.0
        err = sv - self.getObservation()[0] 
        return np.abs(err) # (,)
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.time = 0
        self._state = self.getInitialState()
        self._episode_ended = False
        
        return ts.restart(self.getObservation())

    def _step(self, action):
        # action: (1,)
        
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        if self.time < self.nStepSimulation:
            
            self._state = (1-1/self.T) * self._state + self.K/self.T * action[0]
            
            self.time += 1
            return ts.transition(self.getObservation(), reward = self.getReward(), discount = self.discount)
        else:
            self._episode_ended = True
            return ts.termination(self.getObservation(), reward = self.getReward())


# In[ ]:


def createInstanceOfEnvironment(nStepSimulation = 100):
    return MyEnv(nStepSimulation = nStepSimulation)


# In[ ]:


def runEnvironmentUnitTests():
    
    def aSimpleUnitTest():
        env = createInstanceOfEnvironment()
        assert isinstance(env, py_environment.PyEnvironment)
        utils.validate_py_environment(env, episodes=5)

    def anotherSimpleUnitTest():
        env = createInstanceOfEnvironment()
        assert isinstance(env, py_environment.PyEnvironment)

        u = np.array(np.random.randn(1), np.float32) # (,)

        time_step = env.reset()    
        rewardAvg = time_step.reward    
        while not time_step.is_last():
            time_step = env.step(u)
            rewardAvg = (1-1/10) * rewardAvg + 1/10 * time_step.reward

    aSimpleUnitTest()
    anotherSimpleUnitTest()


# In[ ]:


runEnvironmentUnitTests()


# ## 2. Represent P-controllers by deterministic policy networks or stochastic ones
# 
# MyActionNetDeterminisitc and MyActionNetDistiributional are implementations of P-controller with saturated/bounded outputs.

# In[ ]:


class MyActionNetDeterministic(network.Network):

    def __init__(self, input_tensor_spec, output_tensor_spec):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name='ActionNet')
        self._output_tensor_spec = output_tensor_spec
        self._sub_layers = [
            tf.keras.layers.Dense(
                output_tensor_spec.shape.num_elements(), activation=tf.nn.tanh),
        ]
        self._layer = tf.keras.layers.Dense(output_tensor_spec.shape.num_elements(), activation=tf.nn.tanh)
        # output_tensor_spec
        # BoundedTensorSpec(shape=(3,), dtype=tf.float32, name=None, minimum=array(-1., dtype=float32), maximum=array(1., dtype=float32))

    def call(self, observations, step_type, network_state):
        del step_type

        _observations = tf.cast(observations, dtype=tf.float32) # (nPv,)
        _actions = self._layer(_observations) # (nMv,)
        _actions = tf.reshape(_actions, [-1] + self._output_tensor_spec.shape.as_list()) # (1, nMv)

        return _actions, network_state


# In[ ]:


class MyActionNetDistributional(network.Network):
    """
    
    An instance as stochastic policy represents a P-controller with a random value generator.
    
    >> create an instance of the network:
    net = MyActionNetDistributional(input_tensor_spec, output_tensor_spec)    
    
    """

    def __init__(self, input_tensor_spec, output_tensor_spec):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name='ActionNet')
        self._output_tensor_spec = output_tensor_spec
        self._sub_layers = [
            tf.keras.layers.Dense(
                output_tensor_spec.shape.num_elements(), activation=tf.nn.tanh),
        ]
        self._layer = tf.keras.layers.Dense(output_tensor_spec.shape.num_elements(), activation=tf.nn.tanh)
        self._log_action_std = tf.Variable(tf.zeros(shape=())) # (,)
        # output_tensor_spec
        # BoundedTensorSpec(shape=(3,), dtype=tf.float32, name=None, minimum=array(-1., dtype=float32), maximum=array(1., dtype=float32))

    def call(self, observations, step_type, network_state):
        del step_type

        _observations = tf.cast(observations, dtype=tf.float32) # (nPv,)
        _actions = self._layer(_observations) # (nMv,)
        _actions = tf.reshape(_actions, [-1] + self._output_tensor_spec.shape.as_list()) # (1, nMv)        
        _action_std = tf.ones_like(_actions) * tf.math.exp(self._log_action_std) # (1, nMv)

        return tfp.distributions.MultivariateNormalDiag(_actions, _action_std), network_state


# In[ ]:


def createInstanceOfActorNetwork(input_tensor_spec, output_tensor_spec, isStochastic = True):
    if isStochastic:
        return MyActionNetDistributional(input_tensor_spec, output_tensor_spec)
    else:
        return MyActionNetDeterministic(input_tensor_spec, output_tensor_spec)


# In[ ]:


def runPolicyUnitTests():
    nPv = 1
    nMv = 1
    batch_size = 2**5

    input_tensor_spec = tensor_spec.TensorSpec((nPv,)
                                               , tf.float32)

    action_spec = tensor_spec.BoundedTensorSpec((nMv,),
                                                tf.float32,
                                                minimum=-1,
                                                maximum=1)

    
    for actor_net in (createInstanceOfActorNetwork(input_tensor_spec, action_spec, isStochastic=True)
                            ,createInstanceOfActorNetwork(input_tensor_spec, action_spec, isStochastic=False)):

        my_actor_policy = actor_policy.ActorPolicy(
            time_step_spec = ts.time_step_spec(input_tensor_spec),
            action_spec    = action_spec,
            actor_network  = actor_net)

        observations = tf.random.normal(shape=(batch_size, nPv))

        time_step = ts.restart(observations, batch_size) # time_step.is_first = True

        action_step = my_actor_policy.action(time_step) # action_step.action: (*, nMv)

        distribution_step = my_actor_policy.distribution(time_step)

        assert isinstance(distribution_step.action, tfp.distributions.Distribution)


# In[ ]:


runPolicyUnitTests()


# # 3. Apply an algorithm of RL to design controllers for FOD systems
# 
# See [this tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/6_reinforce_tutorial.ipynb)

# ## 3.1 Implement a factory method of sac agents *TMP*
# see also [this tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/7_SAC_minitaur_tutorial.ipynb)

# hyperparameters

# In[ ]:


env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}


# In[ ]:


from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.train.utils import train_utils
from tf_agents.agents.sac import sac_agent


# In[ ]:


train_py_env = createInstanceOfEnvironment()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)


# In[ ]:


observation_spec, action_spec, time_step_spec = train_env.observation_spec(), train_env.action_spec(), train_env.time_step_spec()


# In[ ]:


critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params,
    kernel_initializer='glorot_uniform',
    last_kernel_initializer='glorot_uniform')


# In[ ]:


actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))


# In[ ]:


train_step = train_utils.create_train_step()

tf_agent = sac_agent.SacAgent(
    time_step_spec,
    action_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=critic_learning_rate),
    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    train_step_counter=train_step)

tf_agent.initialize()


# In[ ]:


def createInstanceOfTfAgent(train_env, learning_rate = 1e-3):
    return tf_agent


# ## 3.1 Implement a factory method of REINFORCE agents
# 
# <font color="red">TO REVISE</font>

# In[ ]:


# def createInstanceOfTfAgent(train_env, learning_rate = 1e-3):
    
#     actor_network = createInstanceOfActorNetwork(input_tensor_spec = train_env.observation_spec()
#                                                            , output_tensor_spec = train_env.action_spec())
    
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#     train_step_counter = tf.compat.v2.Variable(0)

#     tf_agent = reinforce_agent.ReinforceAgent(
#         train_env.time_step_spec(),
#         train_env.action_spec(),
#         actor_network=actor_network,
#         optimizer=optimizer,
#         normalize_returns=True,
#         train_step_counter=train_step_counter)
#     tf_agent.initialize()
    
#     return tf_agent


# In[ ]:


# def runTfAgentUnitTests():
#     train_py_env = createInstanceOfEnvironment()
#     train_env = tf_py_environment.TFPyEnvironment(train_py_env)
#     createInstanceOfTfAgent(train_env)


# In[ ]:


# runTfAgentUnitTests()


# ## 3.2 Implement a process collecting trajectories **TMP**
# 
# see [this tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/5_replay_buffers_tutorial.ipynb)

# In[ ]:


from tf_agents.drivers import dynamic_step_driver


# In[ ]:


train_py_env = createInstanceOfEnvironment()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
tf_agent = createInstanceOfTfAgent(train_env)


# In[ ]:


def createAnInstanceOfReplayBuffer(data_spec, batch_size = 1, replay_buffer_capacity = 1000):

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec,
        batch_size,
        max_length=replay_buffer_capacity)
    
    return replay_buffer


# See [this document](https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers/dynamic_step_driver/DynamicStepDriver)

# In[ ]:


def collectTrajectories(train_env, policy, replay_buffer, collect_steps_per_iteration = 1):
    # Add an observer that adds to the replay buffer:
    replay_observer = [replay_buffer.add_batch]
    
    train_env.reset()

    collect_op = dynamic_step_driver.DynamicStepDriver(
        train_env,
        policy,
        observers=replay_observer,
        num_steps=collect_steps_per_iteration).run()


# In[ ]:


replay_buffer= createAnInstanceOfReplayBuffer(data_spec = tf_agent.collect_data_spec, batch_size=train_env.batch_size)


# In[ ]:


replay_buffer.clear()

nStep = 2**3
collectTrajectories(train_env, tf_agent.collect_policy, replay_buffer, collect_steps_per_iteration = nStep)
iter(replay_buffer.as_dataset(sample_batch_size=1, num_steps=nStep)).__next__()
pass


# ## 3.2 Implement a process collecting trajectories
# 
# <font color="red">TO REVISE</font>
# 
# + [Using EpisodicReplayBuffer in TF-Agents](https://stackoverflow.com/questions/65397939/using-episodicreplaybuffer-in-tf-agents)
# + [episodic_replay_buffer.py](https://github.com/tensorflow/agents/blob/master/tf_agents/replay_buffers/episodic_replay_buffer.py)

# In[ ]:


# def createAnInstanceOfReplayBuffer(data_spec, max_length=2**10):
#     return episodic_replay_buffer.EpisodicReplayBuffer(data_spec
#                                                    , completed_only = True
#                                                    , capacity=max_length) 


# In[ ]:


# def collectTrajectories(environment, policy, replay_buffer, policy_state = (), collect_episodes_per_iteration = 1):
#     replay_buffer.clear()    
#     for iEps in range(collect_episodes_per_iteration):    
#         time_step = environment.reset()    
#         while not time_step.is_last():
#             action_step = policy.action(time_step, policy_state)
#             next_time_step = environment.step(action_step)
#             traj = trajectory.Trajectory(
#                 time_step.step_type,
#                 time_step.observation,
#                 action_step.action,
#                 action_step.info,
#                 next_time_step.step_type,
#                 next_time_step.reward,
#                 next_time_step.discount)

#             replay_buffer.add_batch(traj, tf.constant((iEps,), dtype=tf.int64))

#             time_step = next_time_step
#             policy_state = action_step.state


# In[ ]:


# def runReplayBufferUnitTests():
    
#     nStepSimulation = 3

#     for _ in range(10):
        
#         train_env = tf_py_environment.TFPyEnvironment(createInstanceOfEnvironment(nStepSimulation = nStepSimulation))

#         tf_agent = createInstanceOfTfAgent(train_env)

#         replay_buffer = createAnInstanceOfReplayBuffer(data_spec = tf_agent.collect_data_spec)

#         collect_policy = tf_agent.collect_policy
    
#         collectTrajectories(environment = train_env
#                             , policy = collect_policy
#                             , replay_buffer =  replay_buffer)

#         iterator = iter(replay_buffer.as_dataset(sample_batch_size=1, num_steps = nStepSimulation+1))        
#         trajectories, _ = next(iterator)

#         assert trajectories.is_last().numpy()[0, -1], trajectories.is_last().numpy()
#         assert np.all(~trajectories.is_last().numpy()[0, :-1])


# In[ ]:


# runReplayBufferUnitTests()


# ## 3.3 Implement a process training agents

# In[ ]:


def runTraining(train_env, tf_agent, nStepSimulation, num_iterations):
    
    collect_policy = tf_agent.collect_policy
    
    replay_buffer = createAnInstanceOfReplayBuffer(data_spec = tf_agent.collect_data_spec)
    replay_buffer.clear()
    
    train_env.reset()
    
    tf_agent.train_step_counter.assign(0)

    for _ in range(num_iterations):

        collectTrajectories(train_env= train_env
                            , policy = collect_policy
                            , replay_buffer = replay_buffer
                            , collect_steps_per_iteration = nStepSimulation)

        iterator = iter(replay_buffer.as_dataset(sample_batch_size=2**5, num_steps = nStepSimulation))        
        trajectories, _ = next(iterator)
        train_loss = tf_agent.train(experience=trajectories)


# In[ ]:


def runTrainingUnitTests():
    num_iterations = 3
    nStepSimulation = 2
    train_env = tf_py_environment.TFPyEnvironment(createInstanceOfEnvironment(nStepSimulation = nStepSimulation))
    tf_agent = createInstanceOfTfAgent(train_env)

    
    runTraining(train_env, tf_agent, nStepSimulation, num_iterations)


# In[ ]:


runTrainingUnitTests()

