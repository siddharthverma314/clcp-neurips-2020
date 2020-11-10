from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
import os
import io
from absl import flags, logging
import functools

import sys

sys.path.append(os.path.abspath("./"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import suite_mujoco
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.trajectories.trajectory import from_transition, to_transition
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import ou_noise_policy, policy_saver
from tf_agents.trajectories import policy_step

# from tf_agents.policies import py_tf_policy
# from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils

from dads import dads_agent

from dads.envs import skill_wrapper
from dads.envs import video_wrapper
from dads.envs.gym_mujoco import ant
from dads.envs.gym_mujoco import half_cheetah
from dads.envs.gym_mujoco import humanoid
from dads.envs.gym_mujoco import point_mass

from dads.envs import dclaw
from dads.envs import dkitty_redesign
from dads.envs import hand_block

from dads.lib import py_tf_policy
from dads.lib import py_uniform_replay_buffer

import dads.dads_off as do

FLAGS = do.FLAGS

# global variables for this script
observation_omit_size = 0
goal_coord = np.array([10.0, 10.0])
sample_count = 0
iter_count = 0
episode_size_buffer = []
episode_return_buffer = []


def main(_):
    # setting up
    start_time = time.time()
    tf.compat.v1.enable_resource_variables()
    tf.compat.v1.disable_eager_execution()
    logging.set_verbosity(logging.INFO)
    global observation_omit_size, goal_coord, sample_count, iter_count, episode_size_buffer, episode_return_buffer

    root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
    if not tf.io.gfile.exists(root_dir):
        tf.io.gfile.makedirs(root_dir)
    log_dir = os.path.join(root_dir, FLAGS.environment)

    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
    save_dir = os.path.join(log_dir, "models")
    if not tf.io.gfile.exists(save_dir):
        tf.io.gfile.makedirs(save_dir)

    print("directory for recording experiment data:", log_dir)

    # in case training is paused and resumed, so can be restored
    try:
        sample_count = np.load(os.path.join(log_dir, "sample_count.npy")).tolist()
        iter_count = np.load(os.path.join(log_dir, "iter_count.npy")).tolist()
        episode_size_buffer = np.load(
            os.path.join(log_dir, "episode_size_buffer.npy")
        ).tolist()
        episode_return_buffer = np.load(
            os.path.join(log_dir, "episode_return_buffer.npy")
        ).tolist()
    except:
        sample_count = 0
        iter_count = 0
        episode_size_buffer = []
        episode_return_buffer = []

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        os.path.join(log_dir, "train", "in_graph_data"), flush_millis=10 * 1000
    )
    train_summary_writer.set_as_default()

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(True):
        # environment related stuff
        env = do.get_environment(env_name=FLAGS.environment)
        py_env = wrap_env(
            skill_wrapper.SkillWrapper(
                env,
                num_latent_skills=FLAGS.num_skills,
                skill_type=FLAGS.skill_type,
                preset_skill=None,
                min_steps_before_resample=FLAGS.min_steps_before_resample,
                resample_prob=FLAGS.resample_prob,
            ),
            max_episode_steps=FLAGS.max_env_steps,
        )

        # all specifications required for all networks and agents
        py_action_spec = py_env.action_spec()
        tf_action_spec = tensor_spec.from_spec(
            py_action_spec
        )  # policy, critic action spec
        env_obs_spec = py_env.observation_spec()
        py_env_time_step_spec = ts.time_step_spec(
            env_obs_spec
        )  # replay buffer time_step spec
        if observation_omit_size > 0:
            agent_obs_spec = array_spec.BoundedArraySpec(
                (env_obs_spec.shape[0] - observation_omit_size,),
                env_obs_spec.dtype,
                minimum=env_obs_spec.minimum,
                maximum=env_obs_spec.maximum,
                name=env_obs_spec.name,
            )  # policy, critic observation spec
        else:
            agent_obs_spec = env_obs_spec
        py_agent_time_step_spec = ts.time_step_spec(
            agent_obs_spec
        )  # policy, critic time_step spec
        tf_agent_time_step_spec = tensor_spec.from_spec(py_agent_time_step_spec)

        if not FLAGS.reduced_observation:
            skill_dynamics_observation_size = (
                py_env_time_step_spec.observation.shape[0] - FLAGS.num_skills
            )
        else:
            skill_dynamics_observation_size = FLAGS.reduced_observation

        # TODO(architsh): Shift co-ordinate hiding to actor_net and critic_net (good for futher image based processing as well)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            tf_agent_time_step_spec.observation,
            tf_action_spec,
            fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
            continuous_projection_net=do._normal_projection_net,
        )

        critic_net = critic_network.CriticNetwork(
            (tf_agent_time_step_spec.observation, tf_action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        )

        if (
            FLAGS.skill_dynamics_relabel_type is not None
            and "importance_sampling" in FLAGS.skill_dynamics_relabel_type
            and FLAGS.is_clip_eps > 1.0
        ):
            reweigh_batches_flag = True
        else:
            reweigh_batches_flag = False

        agent = dads_agent.DADSAgent(
            # DADS parameters
            save_dir,
            skill_dynamics_observation_size,
            observation_modify_fn=do.process_observation,
            restrict_input_size=observation_omit_size,
            latent_size=FLAGS.num_skills,
            latent_prior=FLAGS.skill_type,
            prior_samples=FLAGS.random_skills,
            fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
            normalize_observations=FLAGS.normalize_data,
            network_type=FLAGS.graph_type,
            num_mixture_components=FLAGS.num_components,
            fix_variance=FLAGS.fix_variance,
            reweigh_batches=reweigh_batches_flag,
            skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr,
            # SAC parameters
            time_step_spec=tf_agent_time_step_spec,
            action_spec=tf_action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            target_update_tau=0.005,
            target_update_period=1,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=FLAGS.agent_lr
            ),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=FLAGS.agent_lr
            ),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=FLAGS.agent_lr
            ),
            td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
            gamma=FLAGS.agent_gamma,
            reward_scale_factor=1.0 / (FLAGS.agent_entropy + 1e-12),
            gradient_clipping=None,
            debug_summaries=FLAGS.debug,
            train_step_counter=global_step,
        )

        # evaluation policy
        eval_policy = py_tf_policy.PyTFPolicy(agent.policy)

        # collection policy
        if FLAGS.collect_policy == "default":
            collect_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)
        elif FLAGS.collect_policy == "ou_noise":
            collect_policy = py_tf_policy.PyTFPolicy(
                ou_noise_policy.OUNoisePolicy(
                    agent.collect_policy, ou_stddev=0.2, ou_damping=0.15
                )
            )

        # relabelling policy deals with batches of data, unlike collect and eval
        relabel_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)

        # constructing a replay buffer, need a python spec
        policy_step_spec = policy_step.PolicyStep(
            action=py_action_spec, state=(), info=()
        )

        if (
            FLAGS.skill_dynamics_relabel_type is not None
            and "importance_sampling" in FLAGS.skill_dynamics_relabel_type
            and FLAGS.is_clip_eps > 1.0
        ):
            policy_step_spec = policy_step_spec._replace(
                info=policy_step.set_log_probability(
                    policy_step_spec.info,
                    array_spec.ArraySpec(
                        shape=(), dtype=np.float32, name="action_log_prob"
                    ),
                )
            )

        trajectory_spec = from_transition(
            py_env_time_step_spec, policy_step_spec, py_env_time_step_spec
        )
        capacity = FLAGS.replay_buffer_capacity
        # for all the data collected
        rbuffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
            capacity=capacity, data_spec=trajectory_spec
        )

        if FLAGS.train_skill_dynamics_on_policy:
            # for on-policy data (if something special is required)
            on_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
                capacity=FLAGS.initial_collect_steps + FLAGS.collect_steps + 10,
                data_spec=trajectory_spec,
            )

        # insert experience manually with relabelled rewards and skills
        agent.build_agent_graph()
        agent.build_skill_dynamics_graph()
        agent.create_savers()

        # saving this way requires the saver to be out the object
        train_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(save_dir, "agent"),
            agent=agent,
            global_step=global_step,
        )
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(save_dir, "policy"),
            policy=agent.policy,
            global_step=global_step,
        )
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(save_dir, "replay_buffer"),
            max_to_keep=1,
            replay_buffer=rbuffer,
        )

        setup_time = time.time() - start_time
        print("Setup time:", setup_time)

        with tf.compat.v1.Session().as_default() as sess:
            eval_policy.session = sess
            eval_policy.initialize(None)
            eval_policy.restore(os.path.join(FLAGS.logdir, "models", "policy"))

            plotdir = os.path.join(FLAGS.logdir, "plots")
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)
            do.FLAGS = FLAGS
            do.eval_loop(eval_dir=plotdir, eval_policy=eval_policy, plot_name="plot")

            # outer_observations = []
            # outer_actions = []
            # for skill in range(FLAGS.num_skills):
            #  py_env = wrap_env(
            #      skill_wrapper.SkillWrapper(
            #          env,
            #          num_latent_skills=FLAGS.num_skills,
            #          skill_type=FLAGS.skill_type,
            #          preset_skill=None,
            #          min_steps_before_resample=FLAGS.min_steps_before_resample,
            #          resample_prob=FLAGS.resample_prob),
            #      max_episode_steps=FLAGS.max_env_steps)

            #  # get the goddamn skills
            #  timestep = py_env.reset()
            #  observations = [timestep.observation]
            #  actions = []
            #  for i in range(200):
            #    action = eval_policy.action(timestep).action
            #    timestep = py_env.step(action)
            #    observations.append(timestep.observation)
            #    actions.append(action)
            #  observations = np.array(observations)
            #  actions = np.array(actions)
            #  outer_observations.append(observations)
            #  outer_actions.append(actions)

            # with open('observations.pkl', 'wb') as f:
            #  pkl.dump(np.array(outer_observations), f)
            # with open('actions.pkl', 'wb') as f:
            #  pkl.dump(np.array(outer_actions), f)

            # import ipdb; ipdb.set_trace()
            # for obs in outer_observations:
            #  plt.cla()
            #  plt.plot(obs[:, 0].flatten(), obs[:, 1].flatten())
            #  plt.xlim(-2, 2)
            #  plt.ylim(-2, 2)
            # plt.savefig(os.path.join(FLAGS.logdir, 'plot.png'))


if __name__ == "__main__":
    tf.compat.v1.app.run(main)
