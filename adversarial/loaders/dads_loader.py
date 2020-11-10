"""Modified from dads_off.py"""
import os

import numpy as np
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.networks import actor_distribution_network
from tf_agents.trajectories import policy_step
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

from dads import dads_agent
from dads.envs import skill_wrapper
from dads.lib import py_tf_policy
from dads import dads_off as do

from ant_hrl_maze.policy import Policy
from pathlib import Path


class DadsPolicy(Policy):
    @staticmethod
    def check(path: str):
        return True

    def __init__(self, flag_file: str, *args, **kwargs):
        logdir = str(Path(flag_file).absolute().parent)
        self.flags = self.make_flags(logdir, flag_file)
        self.load()

    def process_observation(self, observation):
        def _shape_based_observation_processing(observation, dim_idx):
            if len(observation.shape) == 1:
                return observation[dim_idx : dim_idx + 1]
            elif len(observation.shape) == 2:
                return observation[:, dim_idx : dim_idx + 1]
            elif len(observation.shape) == 3:
                return observation[:, :, dim_idx : dim_idx + 1]

        # for consistent use
        if self.flags.reduced_observation == 0:
            return observation

        # x-axis
        if self.flags.reduced_observation in [1, 5]:
            red_obs = [_shape_based_observation_processing(observation, 0)]
        # x-y plane
        elif self.flags.reduced_observation in [2, 6]:
            red_obs = [
                _shape_based_observation_processing(observation, 0),
                _shape_based_observation_processing(observation, 1),
            ]

        if self.flags.reduced_observation in [5, 6, 8]:
            red_obs += [
                _shape_based_observation_processing(
                    observation, observation.shape[1] - idx
                )
                for idx in range(1, 5)
            ]

        if isinstance(observation, np.ndarray):
            input_obs = np.concatenate(red_obs, axis=len(observation.shape) - 1)
        elif isinstance(observation, tf.Tensor):
            input_obs = tf.concat(red_obs, axis=len(observation.shape) - 1)
        return input_obs

    @staticmethod
    def make_flags(logdir: str, flag_file: str):
        with open(flag_file, "r") as f:
            args = {}
            for line in f.read().splitlines():
                if not line or line.startswith("#"):
                    continue

                k, v = line.split("=")
                k = k[2:]  # remove leading "--"

                # hack to parse whether v is a float or not
                if "." in v or ("e-" in v and all(map(str.isnumeric, v.split("e-")))):
                    v = float(v)
                elif v.isnumeric():
                    v = int(v)

                args[k] = v
        args["logdir"] = logdir
        return type("Flags", (), args)

    def load(self):
        # setting up
        tf.compat.v1.enable_resource_variables()
        tf.compat.v1.disable_eager_execution()

        root_dir = os.path.abspath(os.path.expanduser(self.flags.logdir))
        if not tf.io.gfile.exists(root_dir):
            tf.io.gfile.makedirs(root_dir)
        log_dir = os.path.join(root_dir, self.flags.environment)

        if not tf.io.gfile.exists(log_dir):
            tf.io.gfile.makedirs(log_dir)
        save_dir = os.path.join(log_dir, "models")
        if not tf.io.gfile.exists(save_dir):
            tf.io.gfile.makedirs(save_dir)

        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            os.path.join(log_dir, "train", "in_graph_data"), flush_millis=10 * 1000
        )
        train_summary_writer.set_as_default()

        global_step = tf.compat.v1.train.get_or_create_global_step()
        with tf.compat.v2.summary.record_if(True):
            # environment related stuff
            env = do.get_environment(env_name=self.flags.environment)
            py_env = wrap_env(
                skill_wrapper.SkillWrapper(
                    env,
                    num_latent_skills=self.flags.num_skills,
                    skill_type=self.flags.skill_type,
                    preset_skill=None,
                    min_steps_before_resample=self.flags.min_steps_before_resample,
                    resample_prob=self.flags.resample_prob,
                ),
                max_episode_steps=self.flags.max_env_steps,
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
            if self.flags.observation_omission_size > 0:
                agent_obs_spec = array_spec.BoundedArraySpec(
                    (env_obs_spec.shape[0] - self.flags.observation_omission_size),
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

            if not self.flags.reduced_observation:
                skill_dynamics_observation_size = (
                    py_env_time_step_spec.observation.shape[0] - self.flags.num_skills
                )
            else:
                skill_dynamics_observation_size = self.flags.reduced_observation

            # TODO(architsh): Shift co-ordinate hiding to actor_net and critic_net (good for futher image based processing as well)
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                tf_agent_time_step_spec.observation,
                tf_action_spec,
                fc_layer_params=(self.flags.hidden_layer_size,) * 2,
                continuous_projection_net=do._normal_projection_net,
            )

            critic_net = critic_network.CriticNetwork(
                (tf_agent_time_step_spec.observation, tf_action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=(self.flags.hidden_layer_size,) * 2,
            )

            if (
                self.flags.skill_dynamics_relabel_type is not None
                and "importance_sampling" in self.flags.skill_dynamics_relabel_type
                and self.flags.is_clip_eps > 1.0
            ):
                reweigh_batches_flag = True
            else:
                reweigh_batches_flag = False

            agent = dads_agent.DADSAgent(
                # DADS parameters
                save_dir,
                skill_dynamics_observation_size,
                observation_modify_fn=self.process_observation,
                restrict_input_size=self.flags.observation_omission_size,
                latent_size=self.flags.num_skills,
                latent_prior=self.flags.skill_type,
                prior_samples=self.flags.random_skills,
                fc_layer_params=(self.flags.hidden_layer_size,) * 2,
                normalize_observations=self.flags.normalize_data,
                network_type=self.flags.graph_type,
                num_mixture_components=self.flags.num_components,
                fix_variance=self.flags.fix_variance,
                reweigh_batches=reweigh_batches_flag,
                skill_dynamics_learning_rate=self.flags.skill_dynamics_lr,
                # SAC parameters
                time_step_spec=tf_agent_time_step_spec,
                action_spec=tf_action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                target_update_tau=0.005,
                target_update_period=1,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.flags.agent_lr
                ),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.flags.agent_lr
                ),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.flags.agent_lr
                ),
                td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
                gamma=self.flags.agent_gamma,
                reward_scale_factor=1.0 / (self.flags.agent_entropy + 1e-12),
                gradient_clipping=None,
                debug_summaries=self.flags.debug,
                train_step_counter=global_step,
            )

            # evaluation policy
            eval_policy = py_tf_policy.PyTFPolicy(agent.policy)

            # constructing a replay buffer, need a python spec
            policy_step_spec = policy_step.PolicyStep(
                action=py_action_spec, state=(), info=()
            )

            if (
                self.flags.skill_dynamics_relabel_type is not None
                and "importance_sampling" in self.flags.skill_dynamics_relabel_type
                and self.flags.is_clip_eps > 1.0
            ):
                policy_step_spec = policy_step_spec._replace(
                    info=policy_step.set_log_probability(
                        policy_step_spec.info,
                        array_spec.ArraySpec(
                            shape=(), dtype=np.float32, name="action_log_prob"
                        ),
                    )
                )

            # insert experience manually with relabelled rewards and skills
            agent.build_agent_graph()
            agent.build_skill_dynamics_graph()

            with tf.compat.v1.Session().as_default() as sess:
                eval_policy.session = sess
                eval_policy.initialize(None)
                eval_policy.restore(os.path.join(self.flags.logdir, "models", "policy"))
                self.policy = eval_policy

    @property
    def num_skills(self):
        return self.flags.num_skills

    def forward(self, obs, skill):
        step = TimeStep(StepType.MID, 0, 1, np.r_[obs, skill])
        return self.policy.action(step).action

    def backward(self, obs):
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    args = parser.parse_args()

    policy = DadsPolicy(args.logdir)

    env = do.get_environment(env_name="Ant-v4")
    obs = env.reset()

    for skill in range(policy.num_skills):
        s = np.zeros(policy.num_skills)
        s[skill] = 1.0
        print(obs, s)

        for _ in range(50):
            obs, _, _, _ = env.step(policy.forward(obs, s))
