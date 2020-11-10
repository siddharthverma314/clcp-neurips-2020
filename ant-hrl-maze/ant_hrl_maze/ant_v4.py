import numpy as np
from gym.envs.mujoco import ant_v3
import os
import copy


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntEnv(ant_v3.AntEnv):
    def __init__(
        self,
        xml_file=None,
        ctrl_cost_weight=1e-3,
        contact_cost_weight=5e-4,
        healthy_reward=0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 10.0),
        reset_noise_scale=0,
        contact_force_range=(-1.0, 1.0),
        exclude_current_positions_from_observation=False,
        reach_bonus=5,
        flipped_cost=300,
    ):
        self.flipped_cost = flipped_cost
        self.reach_bonus = reach_bonus

        if not xml_file:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "ant.xml")
        super().__init__(
            xml_file=xml_file,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            contact_force_range=contact_force_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_bounded = np.isfinite(state).all() and min_z <= state[2] <= max_z
        is_flipped = self.sim.data.get_geom_xmat("torso_geom")[2, 2] < -0.8
        is_torso_touching_ground = abs(state[2]) < 2.7
        return is_bounded and not (is_flipped and is_torso_touching_ground)

    def step(self, action):
        xy_position_before = self.sim.data.qpos.flat[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.sim.data.qpos.flat[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt

        distance = np.linalg.norm(xy_position_after)
        velocity = np.linalg.norm(xy_velocity)

        reward = self.reach_bonus * np.exp(-(distance ** 2))
        reward -= distance
        if not self.is_healthy:
            reward -= self.flipped_cost

        done = self.done

        observation = self._get_obs()
        info = {
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "velocity": velocity,
            "healthy": 0 if self.is_healthy else 1,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        observations = np.concatenate((position, velocity))
        return observations

    def reset_model(self, x=None, y=None):
        qpos = copy.deepcopy(self.init_qpos)
        qvel = copy.deepcopy(self.init_qvel)

        if self._reset_noise_scale:
            qpos[:2] = np.random.randn(2) * self._reset_noise_scale

        if x:
            qpos[0] = x
        if y:
            qpos[1] = y

        self.set_state(qpos, qvel)
        return self._get_obs()
