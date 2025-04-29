

import numpy as np
from typing import List, Tuple
import yaml
from .world import BaseWorld
from .port_gen import PortGenerate
from .utils import *
from .reachable_sample import ReachableArea
from .setting import *
from .astar_planner import AstarPlanner
from .losplanner import LosPlanner

class LocalInfo():
    def __init__(self):
        self.idx = []
        self.pos = []
        self.size = []
        self.yaw = []
        self.vert = []


class Evader():
    '''
        1. 处理环境信息
        2. 传感信息
        3. 逃跑策略
        4. 运动
    '''
    
    def __init__(self, pos:List):
        
        self.scout_range = 5
        self.robot_radius = 0.1
        self.velocity = 1.6
        self.state = pos
        self.init_state = self.state
        self.modified_goal_pos = [self.state[0]+0.5, self.state[1]-0.5]
        self.robot_memory_pos = {}
        self.robots_info_local_ = LocalInfo()
        self.obs_info_local_ = LocalInfo()
        self.set_init_state()
        
        
    def set_init_state(self):
        self.state = self.init_state
        self.sp = self.state
        self.sp_global = [0,0]
        self.f = 0
        self.vel_array = []
        self.sp_ind = 0
    
    
    # region 处理环境信息
    def get_env_info(self, case, goal_tolerance=0.5):
        self.world = BaseWorld(case=case)
        self.goal_tolerance = goal_tolerance
        self.dt_ = self.world.timestep
        self.ws_ = np.array([self.world.world_bounds_x, self.world.world_bounds_y])
        self.world.find_valid_obs()
        self.obsmap = None 
        self.reachable_area = None
    
    def update_env(self, pursuers):
        if self.obsmap is None:
            self.obsmap = PortGenerate(
                                   obstacles=self.world.valid_obstacles,
                                   boundVert=self.world.valid_boundary_obstacles,
                                   ws=self.ws_,
                                   world=self.world,
                                  )
        
        self.obsmap.gen_gates_set(self, pursuers)
        
        if self.reachable_area is None:
            self.reachable_area = ReachableArea(world = self.world, 
                                                pursuers =  pursuers, 
                                                evader = self, 
                                                pred_step = PRED_STEP, 
                                                w_c = 1, 
                                                show=False)
        
    # endregion
    
    
    # region 传感信息
    def local_info_and_collision_check(self, police_agents): 
        multi_robot_pos_real_ = np.zeros((2, len(police_agents)))
        for iRobot in range(0, len(police_agents)):
            multi_robot_pos_real_[:, iRobot] = police_agents[iRobot].state
        p = self.state
        self.robots_info_local_.idx = []
        self.robots_info_local_.pos = []
        nRobot = multi_robot_pos_real_.shape[1]
        collision_mtx_robot = np.zeros((1,nRobot))

        self.obs_info_local_.vert = []
        self.obs_info_local_.idx = []

        nBoxObs = len(self.world.obstacles)
        collision_mtx_obs = np.zeros((1,nBoxObs))

        k = 0
        for jBoxObs in range(0, nBoxObs):
            vert_j = self.world.obstacles[jBoxObs][:,:]
            vert_j_3d = np.hstack((vert_j, np.ones((vert_j.shape[0], 1))))  # 对每个顶点添加Z坐标为0
            pe_3d = np.append(p, 1).reshape(-1, 1)
            self.obs_info_local_.idx.append(jBoxObs)
            self.obs_info_local_.vert.append(self.world.obstacles[jBoxObs][:,:].T)
            k = k + 1
                
        collision_mtx_obs =  is_in_obstacle(p, self.world.obstacles)
        collision_mtx_obs = collision_mtx_obs.reshape(1,1)
        
        k = 0
        for iRobot in range(0, nRobot): # remove evader
            pt = multi_robot_pos_real_[:, iRobot]
            d_ij = np.linalg.norm(p - pt)
            
            if d_ij < self.scout_range and is_in_sight(p, pt, self.obs_info_local_.vert, self.scout_range):
                self.robots_info_local_.idx.append(iRobot)
                self.robots_info_local_.pos.append(multi_robot_pos_real_[:, iRobot])
                k = k + 1
                if d_ij < 2 * self.robot_radius * 1:
                    collision_mtx_robot[0,iRobot] = 1 
                
        self.collision_mtx_ = np.hstack((collision_mtx_robot, collision_mtx_obs))
    
    def update_memory(self):
        '''
        if idx and pos can be seen, update the memory
        
        check idx pos, if idx can be seen, update the memory
        
        check if the memory is changed
        
        '''
        
        # 标记是否有新增或修改
        added_or_modified = False

        # 遍历当前的机器人信息
        for idx, pos in zip(self.robots_info_local_.idx, self.robots_info_local_.pos):
            # 如果 idx 不在当前记忆中，说明是新增的
            if idx not in self.robot_memory_pos:
                self.robot_memory_pos[idx] = list(pos)
                added_or_modified = True
            else:
                # 如果 idx 已存在，但 pos 发生变化，更新并标记为修改
                if list(pos) != self.robot_memory_pos[idx]:
                    self.robot_memory_pos[idx] = list(pos)
                    added_or_modified = True

        # 检查是否有变化
        if added_or_modified:
            self.robot_memory_pos_change = True
        else:
            self.robot_memory_pos_change = False
    
    # endregion


    # region 逃跑策略
    def select_goal(self):
        self.act_planner(self.world, self.obsmap.cost_map.evader_cost_map) 
        self.find_nearest_free_point(self.world)
        self.find_nearest_inner_point()
        self.obsmap.cost_map.update_cost_maps(self)
        goals = self.obsmap.prepare_evader_goals(self, NP_SEED)
        pe = self.reachable_area.select_goal(self,goals, self.obsmap.cost_map)
        self.goal = pe[:,-1].reshape(2)
    
    # endregion
    
    
    # region 运动
    def sel_stgy(self):
       
        self.modified_goal_pos = self.modfiy_goal_projected(self.goal, self.world,'goal')

        self.act_planner(self.world, self.obsmap.cost_map.evader_cost_map) 
        self.global_planner.astar_special = False
        
        state = self.modfiy_goal_projected(self.state.reshape(2), self.world, 'current state')
        start = state
        end = self.modified_goal_pos
        
        self.global_planner.planning(end[0], end[1],start[0], start[1])
        
        m = self.global_planner.path.shape[0]
        
        if m > 1:
            self.sp_ind = 0
        else:
            self.global_planner.astar_special = True 
            
        self.sp = self.state
            
    def act_planner(self, world, occ_map):
        
        # if self.global_planner == None:
        self.global_planner = AstarPlanner(world=world)
        self.global_planner.process_map(occ_map=occ_map)
        
        self.local_planner = LosPlanner(pos=None, 
                                        goal=None, 
                                        world_config=world.config, 
                                        )
        
        
    def modfiy_goal_projected(self, goal, world, type):
        goal_idx = world.grid_map.points_to_idx(goal.reshape(2,1).T)
        if not world.grid_map.occ_map_obs[goal_idx[0][0], goal_idx[0][1]] == np.inf:
            return goal
        else:
            radius = DISCRETE_SIZE
            max_attempt_per_radius = 100

            while True:
                for _ in range(max_attempt_per_radius):
                    angle = np.random.uniform(0,2*np.pi)
                    smaple_point = goal + radius * np.array([np.cos(angle), np.sin(angle)])
                    
                    smaple_idx = world.grid_map.points_to_idx(smaple_point.reshape(2,1).T)
                    
                    if not world.grid_map.occ_map_obs[smaple_idx[0][0], smaple_idx[0][1]] == np.inf:
                        print(f"evader for {type} from invalid goal={goal} to valid goal={smaple_point}")
                        return smaple_point
                    
                radius += DISCRETE_SIZE
                
                
    def move(self, online_ex_time = 1, goal_tolerance=0.05):
        
        self.prev_state = self.state
        
        if self.global_planner.astar_special:
            cur_idx = self.global_planner.world.grid_map.points_to_idx(self.state.reshape(2,1).T)
            if self.global_planner.world.grid_map.occ_map_obs[cur_idx[0][0], cur_idx[0][1]] == np.inf:
                self.find_nearest_free_point(self.world)
                self.find_nearest_inner_point()
            else:
                pass
        else:
            for _ in range(online_ex_time):
                self.local_online_plan(self.sp_ind, goal_tolerance)
                # if self.sp_ind < self.global_planner.traj_global.shape[0]-1 and norm(self.sp_global - self.sp) >= goal_tolerance:
                if self.sp_ind < self.global_planner.path.shape[0]-1 and np.linalg.norm(np.array(self.sp_global) - np.array(self.sp)) >= goal_tolerance:
                    ref_vel = self.velocity
                    
                    cur_pos = self.local_planner.update_pos(self.sp, self.sp_global, ref_vel)
                    self.state = cur_pos
                else:
                    break
                
            self.find_nearest_free_point(self.world)
            self.find_nearest_inner_point()
        
        if self.state.shape == (2,1):
            raise ValueError("The state should be 1D array, but got 2D array.")
            
            
    def local_online_plan(self, path_id:int, goal_tolerance):
        '''
        los planner: 根据astar生成的path, 找离他最近的并且还未遍历过的点，然后采用目视导航的方式，直接朝着这个点走
        '''
        target_pt = None
        mis_dist = float('inf')
        
        for i in range(path_id+1, self.global_planner.path.shape[0]):
            point = self.global_planner.path[i]
            # 计算点到self.sp的距离
            dist = np.linalg.norm(np.array(point) - np.array(self.sp))
            if dist > goal_tolerance and dist < mis_dist:
                target_pt = point
                mis_dist = dist
                self.sp_global = target_pt
                self.sp_ind = i
                break 
            
            
    def find_nearest_free_point(self, world):
        """
        Check if self.pos_real_ is inside any of the obstacles. 
        If it is, find the nearest point outside the obstacles.
        """
        cur_idx = self.global_planner.world.grid_map.points_to_idx(self.state.reshape(2,1).T)
        if self.global_planner.world.grid_map.occ_map_obs[cur_idx[0][0], cur_idx[0][1]] == np.inf:
            # print("Position is inside an obstacle, finding nearest free point.")
            nearest_point = None

            cur_idx = self.global_planner.world.grid_map.points_to_idx(self.state.reshape(2,1).T)
            
            # 替换一种新的策略:
            dt = world.config['map']['timestep']
            # 方法1：从当前位置开始，沿着路径方向寻找最近的自由点
            if self.sp_ind <= self.global_planner.path.shape[0]:
                
                # 方法2：从当前位置开始，沿着路径方向找第一个点
                for i in range(self.sp_ind, self.global_planner.path.shape[0]):
                    next_position = self.global_planner.path[i]
                    new_idx = self.global_planner.world.grid_map.points_to_idx(next_position.reshape(2,1).T)
                    if not self.global_planner.world.grid_map.occ_map_obs[new_idx[0][0], new_idx[0][1]] == np.inf:
                        nearest_point = next_position
                        break
            
            else:
                nearest_point = self.global_planner.path[-1]
                        
                        
            if nearest_point is not None:
                self.state = nearest_point.reshape(2)
                print(f"\033[93mEvader moved to nearest free point: {self.state}\033[0m")
                
            else:
                print("No free point found.")
        else:
            pass
        
    def find_nearest_inner_point(self):
        """
        Check if self.pos_real_ is outside the environment. 
        If it is, find the nearest point inside the environment.
        """
        boundary = np.array([self.global_planner.world.world_bounds_x, self.global_planner.world.world_bounds_y])
        # 检查当前位置是否在边界之外或在边界上
        flag = False
        while not is_in_boundary(self.state.reshape(2,1).T.ravel().tolist(), boundary):
            nearest_point = None
            cur_idx = self.global_planner.world.grid_map.points_to_idx(self.state.reshape(2,1).T)
            
            # 定义方向：左、右、上、下以及对角线方向
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            
            for direction in directions:
                new_idx = cur_idx + np.array(direction)
                new_idx = np.clip(new_idx, [0, 0], [self.global_planner.world.grid_map.nx - 1, self.global_planner.world.grid_map.ny - 1])
                new_point = self.global_planner.world.grid_map.idx_to_points(new_idx)
                new_point = new_point[0]
                
                # 检查该点是否在边界内
                if is_in_boundary(new_point, self.global_planner.boundary):
                    nearest_point = new_point
                    flag = True
                    self.state = nearest_point.reshape(2)
                    print(f"Evader Moved to nearest inner point: {self.state}")
                    break
            
            if flag:
                break
            
        if not is_in_boundary(self.state.reshape(2,1).T.ravel().tolist(), boundary):
            raise ValueError("No inner point found.")       
    
    
    # endregion

