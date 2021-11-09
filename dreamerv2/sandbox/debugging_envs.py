from collections import OrderedDict
import cv2
import gym
import numpy as np

try:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multiagent'))
    from multiagent.pygame_environment import PGMultiAgentEnv
    import multiagent.scenarios as scenarios
except:
    raise ImportError('Make sure to link multiagent-particle-envs/multiagent as a subdirectory inside sandbox/')

class Balls:
    # have a mapping from names to scenarios here
    scenarios = {
        'whiteball_push': 'intervenable_bouncing_white_action'
    }

    def __init__(self, name, action_repeat=1, size=(64, 64), max_episode_length=200, seed=0, headless=False):
        if headless:
            # import xvfbwrapper
            # vdisplay = xvfbwrapper.Xvfb()
            # vdisplay.start()
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # works for geb

        scenario = scenarios.load(self.scenarios[name] + ".py").Scenario()
        world = scenario.make_world(k=4)
        self._env = PGMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer=True)

        self._action_repeat = action_repeat
        self._size = size
        self.max_episode_length = max_episode_length
        assert self.max_episode_length > 1, 'if you want self.max_episode_length <= 1, make sure we do not have an off-by-one error in determining is_last in self.step()'

        self.action_dim_p = 2 * self._env.world.dim_p
        self.action_dim_c = self._env.world.dim_c
        self.action_dim = self.action_dim_p + 1 + self.action_dim_c

    @property
    def obs_space(self):
        spaces = {
            'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        # TODO: need to check if image is np.uint 8
        return spaces
    
    @property
    def act_space(self):
        # HACKY
        minimum = np.ones(self.action_dim_p) * -2
        maximum = np.ones(self.action_dim_p) * 2
        action = gym.spaces.Box(minimum, maximum, dtype=np.float32)
        return {'action': action}
    
    def step(self, action):
        assert np.isfinite(action['action']).all(), action['action']
        num_entities = len(self._env.world.policy_agents)
        actor_idx = 0  # hardcoded to whiteball_push

        act_n = OrderedDict()
        for i in range(num_entities):
            if i == actor_idx:
                u = np.zeros(1 + self.action_dim_p)
                u[1:] = action['action']
                act_n[i] = np.concatenate([u, np.zeros(self.action_dim_c)])
            else:
                act_n[i] = np.zeros(self.action_dim)

        is_last = False
        is_terminal = False 

        reward = 0.0
        for _ in range(self._action_repeat):
            obs_n, reward_n, done_n, info_n = self._env.step(act_n)
            reward += reward_n[actor_idx]
            self.tick += 1
            if self.tick >= self.max_episode_length:
                is_last = True
                break

        frame = self.render(mode='rgb_array')[0]
        frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)

        obs = {
            'reward': reward,
            'is_first': False,
            'is_last': is_last,
            'is_terminal': is_terminal,
            'image': frame,
        }
        return obs

    def reset(self):
        self.tick = 0
        time_step = self._env.reset()
        frame = self.render(mode='rgb_array')[0]
        frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        obs = {
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'image': frame  # (H, W, C)
        }
        return obs

    def render(self, mode):
        if mode == 'human':
            return self._env.render(mode)
        elif mode == 'rgb_array':
            frame = self._env.viewer.render_uint8(entities=self._env.world.entities, target_size=(64,64))
            # print(frame.dtype)
            # assert False
            return [frame]
        else:
            raise NotImplementedError


class MutedBalls(Balls):
    scenarios = {
        'whiteball_push': 'intervenable_bouncing_white_action',
    }

    @property
    def act_space(self):
        # HACKY
        minimum = np.ones(self.action_dim_p) * -1
        maximum = np.ones(self.action_dim_p)
        action = gym.spaces.Box(minimum, maximum, dtype=np.float32)
        return {'action': action}

class VeryMutedBalls(Balls):
    scenarios = {
        'simple_box4': 'simple_box4_coll_rcolor',
        'simple_box': 'simple_box'
    }

    @property
    def act_space(self):
        # HACKY
        minimum = np.ones(self.action_dim_p) * -0.5
        maximum = np.ones(self.action_dim_p) * 0.5
        action = gym.spaces.Box(minimum, maximum, dtype=np.float32)
        return {'action': action}
