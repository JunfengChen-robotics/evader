
import numpy as np
from .astar_planner import AstarPlanner
from .costmap import CostMap
from .utils import *
from .setting import *


class ReachableArea(): 
    
    def __init__(self, world = None, pursuers = None , evader =None, pred_step=8, w_c = 1.0 ,show=True):
        self.dt = world.timestep 
        self.t_span = np.arange(0, self.dt*pred_step, self.dt)
        self.pursuers = pursuers
        self.evader = evader
        
        self.relate_velocity_coeff = pursuers[0].max_velocity / evader.velocity
        
        self.capture_range = pursuers[0].capture_range * w_c
        self.scope_e = evader.scout_range
        self.robot_radius = pursuers[0].robot_radius
        self.pursuer_v = pursuers[0].max_velocity
        self.evader_v = evader.velocity

        self.obstacles = world.obstacles
        self.world = world
        self.boundary = np.array([self.world.world_bounds_x, self.world.world_bounds_y]) + np.array([1, -1, 1, -1]).reshape(2, 2)* self.robot_radius*1
        
        self.nBoxObs_ = len(self.world.obstacles) 
        self.nPursuer_ = len(pursuers)   
        
        self.obstacles = self.world.obstacles
        self.obstacles_list = convert_to_nested_list(self.obstacles)
        
        self.selected_goal = None
        self.selected_idx = None
        
        self.show_animation = show
        self.color = ['r', 'g', 'b', 'm' , 'c', 'k', 'y']
        self.fig2 = None

        self.capture = False
        # self.outside = False
        self.astar_planner = AstarPlanner(world=self.world)
        
        self.bound_goals = []
        self.tmp_forbidden_goals = []
        self.tmp_forbidden_loc_goals = []
        self.loc_goals_static = []
        self.loc_goals = []
        
        self.cost_map = CostMap(self.obstacles, np.array([self.world.world_bounds_x, self.world.world_bounds_y]))
        
    
    
    def select_goal(self, evader, evader_goals, cost_map):
        loc_p_pos = [np.array(value) for value in evader.robot_memory_pos.values()]

        goals_infos = {tuple(goal): {"score": -np.inf, "evader_dist": -np.inf, "pursuers_dist":[],"next_goal":None} for goal in evader_goals}

        # self.cost_map.update_evader_cost_map(evader) 
        self.cost_map = cost_map

        new_tmp_forbidden_goals = []
        for goal in self.tmp_forbidden_goals:
            if any(np.array_equal(goal, current_goal) for current_goal in evader_goals):
                new_tmp_forbidden_goals.append(goal)

        if len(new_tmp_forbidden_goals) >= 3/4 * len(evader_goals): # reset all
            new_tmp_forbidden_goals = []
            evader.robot_memory_pos = {}

        self.tmp_forbidden_goals = new_tmp_forbidden_goals

        for goal in evader_goals:

            if (np.linalg.norm(evader.state - goal) < 3*self.robot_radius or is_in_obstacle(goal, self.world.grid_map.inflated_obstacles)) \
                  and not any(np.array_equal(goal, forbidden_goal) for forbidden_goal in self.tmp_forbidden_goals):
                self.tmp_forbidden_goals.append(goal)

            if any(np.array_equal(goal, forbidden_goal) for forbidden_goal in self.tmp_forbidden_goals):
                # remove the goal from goals_infos
                goals_infos.pop(tuple(goal))
                continue

            if GOAL_SELECT == "QUICK":
                return goal.reshape(2,1)

            self.astar_planner.process_map(self.cost_map.evader_cost_map)
            evader_pos, evader_dist = self.calculate_evader_pos_obs(evader, goal)

            goals_infos[tuple(goal)]["evader_dist"] = evader_dist
            goals_infos[tuple(goal)]["next_goal"] = evader_pos[-1].reshape(2,1)

            goals_infos[tuple(goal)]["score"] = 1/(evader_dist+1)


        max_goal = max(goals_infos, key=lambda x: goals_infos[x]["score"])

        self.selected_goal = max_goal

        return goals_infos[max_goal]["next_goal"]
    


    def calculate_evader_pos_obs(self, evader, goal):
        
        start = evader.state
        end = goal
        
        self.astar_planner.planning(end[0], end[1],start[0], start[1])
        
        dist_len = len(self.astar_planner.path)
        
        if dist_len > len(self.t_span):
            path = np.array(self.astar_planner.path)[:len(self.t_span)]
        else:
            path = np.array(self.astar_planner.path)

        return path, dist_len

    


def convert_to_nested_list(obstacles):
    nested_list = []
    for obs in obstacles:
        nested_list.append(obs.tolist())
    return nested_list