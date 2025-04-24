import os
from .map import MapGenerate
from .visgraph.vis_graph import VisGraph
import pyvisgraph as vg
import numpy as np
import yaml
from .utils import *
from shapely.geometry import Polygon, LineString, Point 


class BaseWorld():
    
    def __init__(self, config='world.yaml', coord_config='coord.yaml',random_seed= [], case='case1'):
        self.case = case
        self.load_world_config(config)
        self.load_coord_config(coord_config)
        self.set_random_seed(random_seed)
        self.construct_world()
        self.init_visgraph()
        
    
    def load_world_config(self, config):
        path = os.path.join(os.getcwd())
        config_path = os.path.join(path, 'config/', config)
        with open(config_path, "r", encoding='utf-8') as stream:
            try:
                self._config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    
    def load_coord_config(self, coord_config):
        path = os.path.join(os.getcwd())
        config_path = os.path.join(path, 'config/', coord_config)
        with open(config_path, "r") as stream:
            try:
                self._coord_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
                
    def set_random_seed(self,random_seed):
        if random_seed==[]:
            self._random_seed=self.config['global_parameters']['default_randomseed']
        else:
            self._random_seed=random_seed
    
        
    @property
    def config(self) -> dict:
        return self._config
    
    @property
    def coordinates(self) -> dict:
        return self._coord_config

    
    @property
    def random_seed(self)->int:
        return self._random_seed


    @property
    def timestep(self)->int:
        return self.config['map']['timestep']
    
    @property
    def mapsize(self)->int:
        return self.config['map']['size'][self.case]
    
    @property
    def dx(self)->float:
        return self.config['map']['dx'][self.case]
    
    
    def construct_world(self):
        #set numpy random seed when random_seed is not empty
        if self.random_seed != []: 
            np.random.seed(self.random_seed)
            
        # firstly, we build the map
        self.world_bounds_x = [0.0, 2*self.mapsize]
        self.world_bounds_y = [0.0, 2*self.mapsize]

        self.obstacles = [np.array(obstacle) for obstacle in self.coordinates.get(self.case, [])]
        
        self.obstacles_boxVert = np.array(self.obstacles,dtype=object) # 形状是不一样
        # self.obstacles = self.mouse_map()
        
        ws_ = np.array([self.world_bounds_x, self.world_bounds_y])
        
        self.grid_map = MapGenerate(obstacles=self.obstacles, ws=ws_)
        self.occ_map_obs = self.grid_map.occ_map_obs
        self.occ_map_evader = self.grid_map.occ_map_obs
    

    def init_visgraph(self):
        # using obstacles to initialize the visibility graph
        self.visgraph = VisGraph()
        boundary = [[self.world_bounds_x[0], self.world_bounds_x[1]], [self.world_bounds_y[0], self.world_bounds_y[1]]]
        Polys = []
        for obstacle in self.obstacles:
            # obstacle = self.inflate_obstacles(obstacle, 0.1)
            Polys.append([vg.Point(p[0], p[1]) for p in obstacle]) # 为了构建visgraph，需要将障碍物转换为点
            map_vertices = [(self.world_bounds_x[0], self.world_bounds_y[0]),
                            (self.world_bounds_x[1], self.world_bounds_y[0]),
                            (self.world_bounds_x[1], self.world_bounds_y[1]),
                            (self.world_bounds_x[0], self.world_bounds_y[1])
                            ]
            for point in map_vertices:
                Polys.append([vg.Point(point[0], point[1])])

        self.visgraph.build(Polys, workers=1, boundary=boundary)
        
        
    def find_valid_obs(self):
        '''
        由于我们是多障碍物环境，实则除了evader周围的障碍物，其他障碍物并且离得比较远的都是无用的
        采用的方法是： 为每个障碍物生成三个参考点(中心点，两个最远点)，然后判断这个障碍物是否和其他障碍物相交
        如果有其中的两个点相交，则这个障碍物是无效的
        
        *** important****
        将边界考虑进行

        Updated by Yinhang:

        Hard constraints:
        1. If the obstacle is too far from the evader, ignore the obstacle (distance > 1/2 * sqrt(2*max_length**2))
        2. If the boundary projection point is too far from the obstacle, ignore the projection point (distance > 1/2 * max_length)
        3. If point projection intersects with other obstacles, ignore the projection point

        '''
        self.valid_obstacles = []
        self.valid_boundary_obstacles_candidate= []

        # add map vertices to valid_boundary_vertices
        self.valid_boundary_obstacles_candidate.extend([
            np.array([[self.world_bounds_x[0], self.world_bounds_y[0]]]),  
            np.array([[self.world_bounds_x[1], self.world_bounds_y[0]]]),
            np.array([[self.world_bounds_x[1], self.world_bounds_y[1]]]),
            np.array([[self.world_bounds_x[0], self.world_bounds_y[1]]])
        ])

        if not self.obstacles:
            return

        # 形成四条边界线
        map_vertices = [
            (self.world_bounds_x[0], self.world_bounds_y[0]),
            (self.world_bounds_x[1], self.world_bounds_y[0]),
            (self.world_bounds_x[1], self.world_bounds_y[1]),
            (self.world_bounds_x[0], self.world_bounds_y[1])
        ]
        self.map_edges = [(map_vertices[i], map_vertices[(i+1)%4]) for i in range(4)]

        max_length = max(self.world_bounds_x[1] - self.world_bounds_x[0], self.world_bounds_y[1] - self.world_bounds_y[0])

        self.boundary_obstales = {}
        self.obstalces_edges = {}

        # 先检查障碍物的有效性
        # self.valid_obstacles = self.check_valid_obstacles(pe, self.obstacles, self.obstacles, max_length)
        self.valid_obstacles = self.obstacles

        skeleton_points = [point for obstacle in self.valid_obstacles for point in find_skeleton(obstacle)]

        # 接着处理边界投影
        
        proj_edges = {}
        
        for ob_id, obstacle in enumerate(self.obstacles):
            if self.array_in_list(obstacle, self.valid_obstacles):
                self.boundary_obstales[ob_id] = []

                skeleton = find_skeleton(obstacle)

                # min_x, max_x = np.min(skeleton[:, 0]), np.max(skeleton[:, 0])
                # min_y, max_y = np.min(skeleton[:, 1]), np.max(skeleton[:, 1])

                # rectangle_vertices = [np.array([min_x, min_y]), np.array([max_x, min_y]), np.array([max_x, max_y]), np.array([min_x, max_y])]

                for cand_pt in skeleton:
                    for i in range(4):
                        bound_edge = self.map_edges[i]

                        projection_point = project_point_to_line(cand_pt, bound_edge)

                        intersect = False

                        if np.linalg.norm(cand_pt - projection_point) >= 1/2 * max_length:
                            continue   # ignore the projection point if it is too far from the obstacle
                        
                        if self.array_in_list(np.array(projection_point), skeleton_points):
                            continue

                        # proj_mid = (cand_pt + projection_point) / 2
                        # proj_radius = np.linalg.norm(cand_pt - proj_mid)

                        # proj_circle = Point(proj_mid[0], proj_mid[1]).buffer(proj_radius)

                        # for point in skeleton_points:
                        #     if np.array_equal(point, cand_pt):
                        #         continue
                        #     if proj_circle.contains(Point(point[0], point[1])):
                        #         intersect = True
                        #         break
                        
                        proj_length = np.linalg.norm(cand_pt - projection_point)
                        if cand_pt[0] == projection_point[0]:  # vertical line
                            pt1 = [cand_pt[0]-proj_length/2, cand_pt[1]]
                            pt2 = [cand_pt[0]+proj_length/2, cand_pt[1]]
                            pt3 = [cand_pt[0]-proj_length/2, projection_point[1]]
                            pt4 = [cand_pt[0]+proj_length/2, projection_point[1]]
                        else:
                            pt1 = [cand_pt[0], cand_pt[1]-proj_length/2]
                            pt2 = [cand_pt[0], cand_pt[1]+proj_length/2]
                            pt3 = [projection_point[0], cand_pt[1]-proj_length/2]
                            pt4 = [projection_point[0], cand_pt[1]+proj_length/2]
                            
                        sorted_pts = [pt1, pt2, pt4, pt3]
                        
                        for point in skeleton_points:
                            if np.array_equal(point, cand_pt) or np.array_equal(point, projection_point):
                                continue
                            else:
                                if Polygon(sorted_pts).contains(Point(point[0], point[1])):
                                    intersect = True
                                    break

                        if not intersect:
                            if not self.array_in_list(np.array([projection_point]), self.valid_boundary_obstacles_candidate):
                                
                                current_length = proj_length
                                current_line = LineString([cand_pt, projection_point])
                                
                                # check intersection with other proj edges
                                for proj_pt, proj_edge in proj_edges.items():
                                    if proj_edge["valid"] and proj_edge["lineString"].intersects(current_line) and \
                                        (len(set(proj_edge["lineString"].coords) & set(current_line.coords)) != 1):
                                        # check length, set the longer one to invalid
                                        if proj_edge["length"] > current_length:
                                            proj_edge["valid"] = False
                                        elif proj_edge["length"] <= current_length:
                                            proj_edges[tuple(projection_point)] = {"valid": False, "lineString": current_line, "length": current_length}
                                            break
                                else:
                                    proj_edges[tuple(projection_point)] = {"valid": True, "lineString": current_line, "length": current_length}

        self.valid_boundary_obstacles_candidate.extend([np.array([point]) for point in proj_edges if proj_edges[point]["valid"]])
        
        self.valid_boundary_obstacles = self.valid_boundary_obstacles_candidate
        
        
    def array_in_list(self, array, lst):
        return any(np.array_equal(array, x) for x in lst)   
