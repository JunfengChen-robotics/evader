import numpy as np
from shapely.geometry import Point, Polygon
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from .setting import *
from tkinter import TclError


def default_index_map_value():
    return {'diag': [], 'non_diag': []}

class CostMap:
    
    def __init__(self, obstacles=None,ws = None):
        self.obstacles = obstacles
        self.inflate_obstacles = self.inflate_obstacles(obstacles)
        self.boundary = ws
        self.xy_res = DISCRETE_SIZE
        self.obs_base, self.obs_sigma = 10, 1

        self.ncol = int(np.ceil(np.abs(ws[0, 1] - ws[0, 0]) / self.xy_res))
        self.nrow = int(np.ceil(np.abs(ws[1, 1] - ws[1, 0]) / self.xy_res))

        self._init_grid_and_idx()
        self._init_cost_maps()

    def _init_grid_and_idx(self):

        start_time = time.time()
        
        self.grid = np.zeros((self.nrow, self.ncol))  # Grid indexed by [row, col]
        self.vertex_coords = []
        
        for obs in self.obstacles:
            obs_polygon = Polygon(obs)
            for col in range(self.ncol):  # Iterate over columns (x-direction)
                for row in range(self.nrow):  # Iterate over rows (y-direction)
                    # if row == 0 or col == 0 or row == self.nrow - 1 or col == self.ncol - 1:
                    #     self.grid[row, col] = np.inf
                    #     continue
                    x_coord = col * self.xy_res + 0.5 * self.xy_res
                    y_coord = row * self.xy_res + 0.5 * self.xy_res
                    # if obs_polygon.contains(Point(x_coord, y_coord)) or obs_polygon.intersects(Point(x_coord, y_coord)):
                    if obs_polygon.contains(Point(x_coord, y_coord)):
                        self.grid[row, col] = np.inf

                    elif obs_polygon.intersects(Point(x_coord, y_coord)):
                        self.grid[row, col] = MAIN_BASE

            for vertex in obs:
                if vertex[0] == self.boundary[0, 0] or vertex[0] == self.boundary[0, 1] or vertex[1] == self.boundary[1, 0] or vertex[1] == self.boundary[1, 1]:
                    continue
                vertex_grid_pos = (
                    int(np.floor(vertex[1] / self.xy_res)),  # Row (y)
                    int(np.floor(vertex[0] / self.xy_res))  # Column (x)
                )
                self.vertex_coords.append(vertex_grid_pos)

        self.index_map = defaultdict(default_index_map_value)

        for row in range(self.nrow):
            for col in range(self.ncol):
                non_diag, diag = self._get_neighbours(row, col)
                self.index_map[(row, col)]['diag'] = diag
                self.index_map[(row, col)]['non_diag'] = non_diag

        end_time = time.time()

        # print('Time taken to initialize grid and index map:', end_time - start_time)

    def _get_neighbours(self, row, col):  # row and col represent the position in the grid
        diagonal_neighbours = []
        non_diagonal_neighbours = []

        for drow in [-1, 0, 1]:
            for dcol in [-1, 0, 1]:
                if drow == 0 and dcol == 0:
                    continue
                new_row = row + drow
                new_col = col + dcol
                if 0 <= new_row < self.nrow and 0 <= new_col < self.ncol:
                    if drow * dcol == 0:
                        non_diagonal_neighbours.append((new_row, new_col))
                    else:
                        diagonal_neighbours.append((new_row, new_col))

        return non_diagonal_neighbours, diagonal_neighbours

    def _init_cost_maps(self):

        self.evader_cost_map = deepcopy(self.grid).T
        self.init_evader_cost_map_pre()
        self.closure_polys_map = None

    def add_action_to_map(self, closure_polys, push_gate):
        self.update_closure_polys_map(closure_polys)
        self.push_gate = push_gate

    def update_closure_polys_map(self, closure_polys):
        self.closure_polys_map = np.zeros((self.nrow, self.ncol))

        for col in range(self.ncol):  # Iterate over columns (x-direction)
            for row in range(self.nrow):  # Iterate over rows (y-direction)
                x_coord = col * self.xy_res + 0.5 * self.xy_res
                y_coord = row * self.xy_res + 0.5 * self.xy_res
                if any([Polygon(poly).contains(Point(x_coord, y_coord)) for poly in closure_polys]) \
                        or any([Polygon(poly).intersects(Point(x_coord, y_coord)) for poly in closure_polys]):
                    self.closure_polys_map[row, col] = 1

    def update_cost_maps(self, evader):
        start_time = time.time()

        self.update_hider_cost_map(evader)
        if evader.robot_memory_pos_change:
            self.update_evader_cost_map(evader)
        self.update_attacker_cost_map(evader)

        end_time = time.time()

    def init_evader_cost_map_pre(self):

        self.evader_cost_map_pre = deepcopy(self.grid)

        def add_cost_to_map(comb):
            queue = deque([(pos, base, sigma, 0) for pos, base, sigma in comb])
            visited = set()
            visited.add( pos for pos, _, _ in comb)
            
            while queue:
                pos, base, sigma, distance = queue.popleft()

                guassian_value = base * np.exp(-distance ** 2 / (2 * sigma ** 2))
                if guassian_value < 1:
                    continue

                if self.grid[pos[0], pos[1]] == np.inf:
                    self.evader_cost_map_pre[pos[0], pos[1]] = np.inf

                else:
                    self.evader_cost_map_pre[pos[0], pos[1]] += guassian_value

                for neighbour in self.index_map[pos]['non_diag']:
                    if neighbour not in visited and self.grid[neighbour[0], neighbour[1]] == 0:
                        queue.append((neighbour, base, sigma, distance + 1))
                        visited.add(neighbour)

                for neighbour in self.index_map[pos]['diag']:
                    if neighbour not in visited and self.grid[neighbour[0], neighbour[1]] == 0:
                        queue.append((neighbour, base, sigma, distance + np.sqrt(2)))
                        visited.add(neighbour)

        start_points2 = self.vertex_coords

        start_points3 = [(row, col)
            for row in range(self.nrow)
            for col in range(self.ncol)
            if row == 0 or col == 0 or row == self.nrow - 1 or col == self.ncol - 1]
        
        comb2 = [(start_point, ASSIST_BASE, ASSIST_SIGMA) for start_point in start_points2]
        comb2 += [(start_point, BOUNDARY_BASE, BOUNDARY_SIGMA) for start_point in start_points3]

        add_cost_to_map(comb2)

    def update_evader_cost_map(self, evader):

        # start_time = time.time()

        self.evader_cost_map = deepcopy(self.evader_cost_map_pre)

        pursuer_phys_poses = evader.robot_memory_pos.values()

        pursuer_grid_poses = [
            (
                int(np.floor(pursuer_phys_pos[1] / self.xy_res)),  # Row (y)
                int(np.floor(pursuer_phys_pos[0] / self.xy_res))  # Column (x)
            )
            for pursuer_phys_pos in pursuer_phys_poses
        ]

        def add_cost_to_map(comb):
            queue = deque([(pos, base, sigma, 0) for pos, base, sigma in comb])
            visited = set()
            visited.add( pos for pos, _, _ in comb)
            
            while queue:
                pos, base, sigma, distance = queue.popleft()

                guassian_value = base * np.exp(-distance ** 2 / (2 * sigma ** 2))
                if guassian_value < 1:
                    continue

                # if base == BOUNDARY_BASE:
                #     if distance > np.sqrt(2):
                #         continue

                if self.grid[pos[0], pos[1]] == np.inf:
                    self.evader_cost_map[pos[0], pos[1]] = np.inf

                else:
                    # guassian_value = base * np.exp(-distance **2 / (2 * sigma ** 2))
                    self.evader_cost_map[pos[0], pos[1]] += guassian_value

                for neighbour in self.index_map[pos]['non_diag']:
                    if neighbour not in visited and self.grid[neighbour[0], neighbour[1]] == 0:
                        queue.append((neighbour, base, sigma, distance + 1))
                        visited.add(neighbour)
                        # if base == MAIN_BASE:
                        #     visited.add(neighbour)

                for neighbour in self.index_map[pos]['diag']:
                    if neighbour not in visited and self.grid[neighbour[0], neighbour[1]] == 0:
                        queue.append((neighbour, base, sigma, distance + np.sqrt(2)))
                        visited.add(neighbour)
                        # if base == MAIN_BASE:
                        #     visited.add(neighbour)

        start_points1 = pursuer_grid_poses 
        # start_points2 = [(row, col) 
        #     for row in range(self.nrow)
        #     for col in range(self.ncol)
        #     if self.grid[row, col] == np.inf ]
        start_points2 = self.vertex_coords
        
        start_points3 = [(row, col)
            for row in range(self.nrow)
            for col in range(self.ncol)
            if row == 0 or col == 0 or row == self.nrow - 1 or col == self.ncol - 1]
        
        comb1 = [(start_point, MAIN_BASE, ASSIST_SIGMA_P) for start_point in start_points1]
        comb2 = [(start_point, ASSIST_BASE, ASSIST_SIGMA) for start_point in start_points2]
        comb2 += [(start_point, BOUNDARY_BASE, BOUNDARY_SIGMA) for start_point in start_points3]
        
        add_cost_to_map(comb1)
        # add_cost_to_map(comb2)
        
        self.evader_cost_map = self.evader_cost_map.T

        # end_time = time.time()

        # print('Time taken to update evader cost map:', end_time - start_time)

    def inflate_obstacles(self, obstacles, margin=0.1):
        inflated_obstacles = []
        for box in obstacles:
            polygon = Polygon(box)
            inflated_polygon = polygon.buffer(margin)
            inflated_obstacles.append(np.array(inflated_polygon.exterior.coords))
        return inflated_obstacles

    def calculate_sigma(self,main_base, assist_base, distance):
        sigma = np.sqrt(-max(distance,0.1) ** 2 / (2 * np.log(assist_base / main_base)))
        return sigma
    
    
    def points_to_idx(self, xy=None):
        return self.world.points_to_idx(xy)