import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


class Walker2dFallingEnv(Walker2dEnv):

    def __init__(self):
        self.contact_triggered = False
        self.contact_counter = 0
        self.n_post_contact_frames_till_done = 50
        self.tick_limit = 200
        self.tick = 0
        self.has_reset = False

        mujoco_env.MujocoEnv.__init__(self, "walker2d_cheap.xml", 15)
        utils.EzPickle.__init__(self)

    def step(self, a):
        previous_cfrc_ext = self.sim.data.cfrc_ext.copy()
        ob, reward, done, info = super().step(a)
        if not self.contact_triggered and self.is_contact_happening():
            self.contact_triggered = True
        if self.contact_triggered:
            self.contact_counter += 1

        # compute damage
        body_damage_weights = np.array([0, 1e3, 1e1, 1e1, 1e0, 1e1, 1e1, 1e0])
        body_damage_weights /= np.linalg.norm(body_damage_weights)
        impulses = (self.sim.data.cfrc_ext - previous_cfrc_ext) / self.dt
        reward = np.expm1(-np.sum(np.abs(impulses), axis=1) @ body_damage_weights) + 1
        self.tick += 1
        done = self.contact_counter >= self.n_post_contact_frames_till_done and self.tick < self.tick_limit
        # done = self.tick < self.tick_limit
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def is_contact_happening(self):
        return np.any(np.abs(self.sim.data.cfrc_ext) >= 1)

    def reset_model(self):
        self.contact_triggered = False
        self.contact_counter = 0
        self.tick = 0

        init_qpos = self.init_qpos
        init_qvel = self.init_qvel

        # set height
        init_qpos[1] = np.random.uniform(1.5, 3)

        # set pose to something random
        init_qpos[2:] = self.observation_space.sample()[1:8]

        # set x-velocity
        init_qvel[0] = np.random.normal(0, 2)

        # set joint velocity to something random
        init_qvel[2:8] = self.observation_space.sample()[10:16] / 2

        # init_qpos[3] = 1  # left_hip
        # init_qpos[4] = 1  # left_knee
        # init_qpos[5] = 1  # left_ankle
        # init_qpos[6] = 1  # right_hip
        # init_qpos[7] = 1  # right_knee
        # init_qpos[8] = 1  # right_ankle

        self.set_state(
            init_qpos,
            init_qvel
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
