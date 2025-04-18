import numpy as np


class LosPlanner():
    
    def __init__(self, pos, goal, world_config, robot_config):
        self.start_point = pos
        self.target_point = goal
        self.world_config = world_config
        self.robot_config = robot_config
        
        
    def update_pos(self, sp, target, ref_v):
        # update the position of the robot
        dt = self.world_config['map']['timestep']
        self.target_point = target
        dir_vec = self.target_point - sp 
        dir_vec = dir_vec / max(np.linalg.norm(dir_vec), 0.01)
        cur_pos = sp + dir_vec * dt * ref_v
        
        return cur_pos
    