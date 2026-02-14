import numpy as np
import os

import gymnasium

import pufferlib
from pufferlib.ocean.grid import binding

class Grid(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode='raylib', 
        vision=5,
        num_envs=1024, 
        num_maps=8192, 
        size=-1,
        max_size=20,
        horizon=128, 
        speed=1.0, 
        discretize=True,
        difficulty=0.85,
        count_based_reward_coef=0.0,
        report_interval=128, 
        buf=None,
        seed=0,
    ):
        self.max_size = max_size
        self.vision = vision
        self.horizon = horizon
        self.speed = speed
        self.discretize = discretize
        self.size = size    
        self.count_based_reward_coef = count_based_reward_coef 
        self.obs_size = 2*vision + 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=255, shape=(self.obs_size*self.obs_size + 1,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.difficulty = difficulty
        super().__init__(buf=buf)
        self.float_actions = np.zeros_like(self.actions).astype(np.float32)
        
        self.c_state = binding.shared(
            num_maps=num_maps,
            max_size=max_size, 
            size=size,
            seed=seed,
            difficulty=difficulty,
            count_based_reward_coef=count_based_reward_coef,
            vision=vision,
            speed=speed,
        )
        
        self.c_envs = binding.vec_init(
            self.observations, 
            self.float_actions,
            self.rewards, 
            self.terminals, 
            self.truncations, 
            num_envs, 
            seed,
            state=self.c_state, 
            max_size=max_size, 
            num_maps=num_maps,
            horizon=horizon,
            count_based_reward_coef=count_based_reward_coef,
            vision=vision,
            speed=speed,
        )

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self, overlay=0):
        binding.vec_render(self.c_envs, overlay)
        
    def get_coverage_counts(self):
        """Get the cumulative state visitation counts as a 2D numpy array"""
        return binding.vec_get_coverage_counts(self.c_envs)

    def close(self):
        pass

def test_performance(timeout=10, atn_cache=1024):
    env = Grid(max_size=7, size=-1, num_envs=1, num_maps=1, count_based_reward_coef=1.0, horizon=1000)
    env.reset(0)
    tick = 0

    actions = np.random.randint(0, 5, (atn_cache, 1))
    
    import time
    start = time.time()
    while tick < 100:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1
        print(f'Tick: {tick}')
        print(env.get_coverage_counts())
        print(f'r = {env.rewards}')
        if tick % 20 == 0:
            env.reset(0)
                    
    print(env.observations.shape)
    print(env.observations[:, -1])
    
    #print(f'SPS: {env.num_envs * tick / (time.time() - start)}')

if __name__ == "__main__":
    test_performance()