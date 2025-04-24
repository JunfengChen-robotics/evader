import numpy as np
from .astar_planner import AstarPlanner
from .costmap import CostMap
from .utils import *
from .visgraph.vis_graph import VisGraph
from .visgraph.graph import Point as VisGraphPoint
from .visgraph.graph import array_to_points
from .visgraph.visible_vertices import total_euclidean_distance
from itertools import combinations, islice, permutations, product
from collections import OrderedDict, defaultdict, deque
from shapely.geometry import LineString, Point, Polygon
from .setting import *
import networkx as nx
import matplotlib.pyplot as plt


class PortGenerate():
    
    def __init__(self, obstacles=None, boundVert=None, ws=None, world=None):
        self.boundary = ws
        self.obstacles = obstacles
        self.world = world

        self.boundVert = [np.array(boud).reshape(-1,2) for boud in boundVert]
        boxVert_transposed = obstacles
        self.boxVert_list = [[[float(num) for num in inner_list] for inner_list in outer_list.tolist()] for outer_list in boxVert_transposed] # deprecated

        self.ports_vertex = {}
        self.vertices_ports_dict = {}
        self.current_pursuer_space_node_idxs = None
        self.graph = None

        self.astar_planner = AstarPlanner(self.world)
        self.astar_planner.process_map(self.world.occ_map_obs)

        self.cost_map = CostMap(obstacles, ws)
        
        self.prepare_valid_bbox()
        self.prepare_valid_skeleton_polys()
        self.prepare_edge_dict()
        self.prepare_sub_spaces_set()
        
        self.prepare_nodes_dict()
        
        self.generate_ports_v2()
        
        
    def prepare_valid_bbox(self):
        '''
        aim: prepare the valid bbox and boundary
        '''
        
        total_vert = self.obstacles + self.boundVert
        valid_bbox = {}
        idx_to_point_dict, point_to_idx_dict, point_idx = {}, {}, 0
        max_boxVert_index = len(self.obstacles) -1

        for ibox in range(len(total_vert)):
            valid_bbox[ibox] = {'center':None, 'vertex':None, 'type':None, 'skeleton':None, 'skeleton_point_idxs':[]}
            if ibox <= max_boxVert_index:
                valid_bbox[ibox]['type'] = 'box'
            else:
                valid_bbox[ibox]['type'] = 'boundary'
                
            valid_bbox[ibox]['center'] = np.mean(total_vert[ibox], axis=0)
            valid_bbox[ibox]['vertex'] = total_vert[ibox]
            valid_bbox[ibox]['skeleton'] = find_skeleton(valid_bbox[ibox]['vertex'])

            for idx, point in enumerate(valid_bbox[ibox]['skeleton']):
                if valid_bbox[ibox]['type'] == 'box':
                    if idx != 0 and idx != len(valid_bbox[ibox]['skeleton'])-1:
                        pre_vector = valid_bbox[ibox]['skeleton'][idx] - valid_bbox[ibox]['skeleton'][idx-1]
                        next_vector = valid_bbox[ibox]['skeleton'][idx+1] - valid_bbox[ibox]['skeleton'][idx]

                        pre_vector, next_vector = np.round(pre_vector, 2), np.round(next_vector, 2)

                        # 如果平行则为obs
                        point_type = "obs" if np.cross(pre_vector, next_vector) == 0 else "point"
                        
                    else:
                        point_type = "point"

                elif valid_bbox[ibox]['type'] == 'boundary':
                    point_type = "obs"

                valid_bbox[ibox]['skeleton_point_idxs'].append(point_idx)
                point_to_idx_dict[tuple(point)] = {'point_idx':point_idx, 'ibox':ibox, 'skeleton_idxs':[], 'point_type':point_type}
                idx_to_point_dict[point_idx] = {'point':point, 'ibox':ibox, 'skeleton_idxs':[], 'point_type':point_type}
                point_idx += 1
            
        self.valid_bbox = [valid_bbox[ibox] for ibox in range(len(valid_bbox))]
        self.point_to_idx_dict = point_to_idx_dict
        self.idx_to_point_dict = idx_to_point_dict
        
    
    def prepare_valid_skeleton_polys(self):
        '''
        aim: prepare the valid skeleton polys
        '''
        
        skeleton_polys = []
        skeleton_point_box = []
        skeleton_point_boundary = []

        assist_polys = []

        for ibox in range(len(self.valid_bbox)):
            if self.valid_bbox[ibox]['type'] == 'box':
                for i in range(len(self.valid_bbox[ibox]['skeleton'])-1):
                    skeleton_polys.append([self.valid_bbox[ibox]['skeleton'][i], self.valid_bbox[ibox]['skeleton'][i+1]])
                    skeleton_point_box.append([self.valid_bbox[ibox]['skeleton'][i][0], self.valid_bbox[ibox]['skeleton'][i][1]])
                skeleton_point_box.append([self.valid_bbox[ibox]['skeleton'][-1][0], self.valid_bbox[ibox]['skeleton'][-1][1]])

                for i in range(len(self.valid_bbox[ibox]['skeleton'])):
                    if i == 0 or i == len(self.valid_bbox[ibox]['skeleton'])-1:
                        continue

                    pre_vector = self.valid_bbox[ibox]['skeleton'][i] - self.valid_bbox[ibox]['skeleton'][i-1]
                    next_vector = self.valid_bbox[ibox]['skeleton'][i+1] - self.valid_bbox[ibox]['skeleton'][i]

                    if np.cross(pre_vector, next_vector) != 0: # not parallel
                        pre_point = 0.99 * self.valid_bbox[ibox]['skeleton'][i] + 0.01 * self.valid_bbox[ibox]['skeleton'][i-1]
                        next_point = 0.99 * self.valid_bbox[ibox]['skeleton'][i] + 0.01 * self.valid_bbox[ibox]['skeleton'][i+1]
                        assist_polys.append([pre_point, next_point])
                        
            elif self.valid_bbox[ibox]['type'] == 'boundary':
                skeleton_polys.append([self.valid_bbox[ibox]['skeleton'][0]])
                skeleton_point_boundary.append([self.valid_bbox[ibox]['skeleton'][0][0], self.valid_bbox[ibox]['skeleton'][0][1]])

        # change skeleton_point_box from np.array to tuple
        self.skeleton_point_box = [tuple(point) for point in skeleton_point_box]
        self.skeleton_point_boundary = [tuple(point) for point in skeleton_point_boundary]

        # change assist_polys from np.array to tuple
        self.assist_polys = [[tuple(point) for point in poly] for poly in assist_polys]

        # change skeleton_polys from np.array to tuple
        self.skeleton_polys = [[tuple(point) for point in poly] for poly in skeleton_polys]
        self.skeleton_polys_dict = {idx:{'points':[], 'point_idxs':[]} for idx in range(len(skeleton_polys))}
        for i in range(len(skeleton_polys)):
            self.skeleton_polys_dict[i]['points'] = [tuple(point) for point in skeleton_polys[i]]
            for point in skeleton_polys[i]:
                self.skeleton_polys_dict[i]['point_idxs'].append(self.point_to_idx_dict[tuple(point)]['point_idx'])

        # add skeleton_polys to point_to_idx_dict and idx_to_point_dict
        for idx in self.skeleton_polys_dict.keys():
            for point in self.skeleton_polys_dict[idx]['points']:
                self.point_to_idx_dict[point]['skeleton_idxs'].append(idx)
            for point_idx in self.skeleton_polys_dict[idx]['point_idxs']:
                self.idx_to_point_dict[point_idx]['skeleton_idxs'].append(idx)
        
                
    def prepare_edge_dict(self):
        
        valid_visgraph = VisGraph()
        
        skeleton_polys_vg = []
        
        for idx in self.skeleton_polys_dict:
            skeleton_polys_vg.append([VisGraphPoint(point[0], point[1]) for point in self.skeleton_polys_dict[idx]['points']])
            
        valid_visgraph.build(skeleton_polys_vg)
        
        edges = valid_visgraph.visgraph.edges
        
        edges = [[(edge.p1.x, edge.p1.y), (edge.p2.x, edge.p2.y)] for edge in edges]
        
        def remove_edge_near_boundary(edges, boundary):
            new_edges = []
            for edge in edges:
                if is_intersect_boundary(edge, boundary):
                    if not is_almost_equal(edge[0][0], edge[1][0]) and not is_almost_equal(edge[0][1], edge[1][1]):
                        continue
                new_edges.append(edge)
            return new_edges
        
        edges = remove_edge_near_boundary(edges, self.boundary)
        
        edges = self.remove_overlap_edges(edges)
        
        self.edge_to_idx_dict , self.idx_to_edge_dict = {}, {}
        
        edge_idx = 0
        
        for edge in edges:
            p1 = tuple(edge[0])
            p2 = tuple(edge[1])
            
            # 计算edge_mid，保留两位小数
            edge_mid = round((p1[0] + p2[0]) / 2, 2), round((p1[1] + p2[1]) / 2, 2)
            
            if [p1, p2] in self.skeleton_polys or [p2, p1] in self.skeleton_polys:
                type = "skeleton_edge"
            elif gate_near_boundary([p1, p2], self.boundary) or gate_near_boundary([p2, p1], self.boundary):
                type = "boundary_edge"
            else:
                type = "gate_edge"
            
            self.edge_to_idx_dict[(p1, p2)] = {"edge_idx":edge_idx, "type":type, "mid_point":edge_mid}
            self.idx_to_edge_dict[edge_idx] = {"edge":(p1, p2), "type":type, "mid_point":edge_mid}
            
            edge_idx += 1
        
                
    def prepare_sub_spaces_set(self, show=False):
        '''
        use visgraph to split the space
        
        return:
        dict of sub spaces:
            the vertex of each sub space
            the connect relation between sub spaces
            the shared edge between sub spaces
        '''

        # # find sub spaces
        def get_valid_closures():
            closures = []

            # gate_edge and skeleton_edge need 2 times to build
            # boundary_edge need 1 times to build
            # assist_edge need 0 times to build
            build_times = {}
            for edge_idx, edge_info in self.idx_to_edge_dict.items():
                edge, type = edge_info["edge"], edge_info["type"]
                if type == "boundary_edge":
                    if edge not in build_times:
                        build_times[edge] = 0  # boundary edge is no direction
                else:
                    if edge not in build_times:
                        build_times[edge] = 0  # other edges are directed
                
            comb_num = 3
            while True:
                all_vertices = set()
                for edge, times in build_times.items():
                    if times == 0:
                        all_vertices.add(edge[0])
                        all_vertices.add(edge[1])
                        
                all_vertices = list(all_vertices)
                        
                if not all_vertices:
                    break

                for comb in combinations(all_vertices, comb_num):
                    
                    # sorted_cycle = sorted(comb, key=lambda point: math.atan2(point[1] - np.mean(comb, axis=0)[1], point[0] - np.mean(comb, axis=0)[0]))
                    # poly = Polygon(sorted_cycle)
                    overlap = False
                        
                    # edges = [(sorted_cycle[i], sorted_cycle[(i+1)%len(sorted_cycle)]) for i in range(len(sorted_cycle))]
                    # if any(edge not in build_times.keys() for edge in edges) or any(build_times[edge]!=0 for edge in edges):
                    #     continue

                    closure_edges = []
                    for two_point in combinations(comb, 2):
                        if two_point in build_times:
                            closure_edges.append(two_point)
                    
                    vertex_visited_time = defaultdict(int)
                    for edge in closure_edges:
                        vertex_visited_time[edge[0]] += 1
                        vertex_visited_time[edge[1]] += 1
                            
                    if all([times == 2 for times in vertex_visited_time.values()]) and len(vertex_visited_time) == comb_num:
                        
                        closure = self.build_closure(closure_edges)
                        sorted_closure_edges = [tuple([closure[i], closure[(i+1)%len(closure)]]) for i in range(len(closure))]
                        poly = Polygon(closure)
                        
                        for point in self.skeleton_point_box + self.skeleton_point_boundary:
                            if poly.contains(Point(point)) and not poly.touches(Point(point)):
                                overlap = True
                                break                         

                        if not overlap:
                            closures.append(closure)
                                    
                            for edge in sorted_closure_edges:
                                if self.edge_to_idx_dict[edge]["type"] == "gate_edge" or self.edge_to_idx_dict[edge]["type"] == "skeleton_edge":
                                    build_times[edge] = 1
                                elif self.edge_to_idx_dict[edge]["type"] == "boundary_edge":
                                    build_times[edge] = 1
                                    build_times[edge[::-1]] = 1

                comb_num += 1

            return closures

        sub_spaces = []
        closures = get_valid_closures()
        sub_spaces.extend(closures)

        self.sub_spaces_set = {}

        for idx, sub_space in enumerate(sub_spaces):
            self.sub_spaces_set[idx] = {"origin_info":{}, "connect_info":{"connected_edges_idx":set(), "connected_edge_mid_points":[]}}
            self.sub_spaces_set[idx]["origin_info"]["vertex"] = sub_space
            # self.sub_spaces_set[idx]["origin_info"]["mid_point"] = np.mean(sub_space, axis=0)
            self.sub_spaces_set[idx]["origin_info"]["edges_with_type"] = []
            for i in range(len(sub_space)):
                edge = (sub_space[i], sub_space[(i+1)%len(sub_space)])
                type = self.edge_to_idx_dict[tuple(edge)]["type"]

                self.sub_spaces_set[idx]["origin_info"]["edges_with_type"].append((edge, type))

            gates_mid_points = []
            for edge, type in self.sub_spaces_set[idx]["origin_info"]["edges_with_type"]:
                edge_idx = self.edge_to_idx_dict[edge]["edge_idx"]
                if type == "gate_edge":
                    gates_mid_points.append(self.idx_to_edge_dict[edge_idx]["mid_point"])

            self.sub_spaces_set[idx]["origin_info"]["mid_point"] = np.mean(gates_mid_points, axis=0)

            if len(gates_mid_points) == 0:
                del self.sub_spaces_set[idx]
    
        for idx, sub_space in self.sub_spaces_set.items():
            for edge, type in self.sub_spaces_set[idx]["origin_info"]["edges_with_type"]:
                edge_idx = self.edge_to_idx_dict[edge]["edge_idx"]

                self.sub_spaces_set[idx]["connect_info"]["connected_edges_idx"].add(edge_idx)
                self.sub_spaces_set[idx]["connect_info"]["connected_edge_mid_points"].append(self.idx_to_edge_dict[edge_idx]["mid_point"])
                                
            
    def prepare_nodes_dict(self):

        def check_intersection(edge1, edge2):
            line1 = LineString(edge1)
            line2 = LineString(edge2)
            if line1.intersects(line2) and len(set(line1.coords) & set(line2.coords)) != 1:
                return True
            return False
        
        self.node_to_idx_dict , self.idx_to_node_dict = {}, {}
        
        node_idx = 0
        
        unique_nodes = set()

        for point_idx, point_info in self.idx_to_point_dict.items():
            point = tuple(point_info["point"])
            if point not in unique_nodes:
                node_type = point_info["point_type"]
                self.node_to_idx_dict[point] = {"node_idx":node_idx, "node_type":node_type, "connected_node_idxs":set(), "point_idx":point_idx}
                self.idx_to_node_dict[node_idx] = {"node":point, "node_type":node_type, "connected_node_idxs":set(), "point_idx":point_idx}
                node_idx += 1
                unique_nodes.add(point)

        for sub_space_idx, sub_space in self.sub_spaces_set.items():
            node = tuple(sub_space["origin_info"]["mid_point"])
            if node not in unique_nodes:
                self.node_to_idx_dict[node] = {"node_idx":node_idx, "node_type":"sub_space", "connected_node_idxs":set(), "sub_space_id":sub_space_idx}
                self.idx_to_node_dict[node_idx] = {"node":node, "node_type":"sub_space", "connected_node_idxs":set(), "sub_space_id":sub_space_idx}
                node_idx += 1
                unique_nodes.add(node)
                
        for edge_idx, edge_info in self.idx_to_edge_dict.items():
            node = edge_info["mid_point"]
            if node not in unique_nodes:
                if edge_info["type"] == "gate_edge":
                    self.node_to_idx_dict[node] = {"node_idx":node_idx, "node_type":"gate", "connected_node_idxs":set(), "edge_idx":edge_idx}
                    self.idx_to_node_dict[node_idx] = {"node":node, "node_type":"gate", "connected_node_idxs":set(), "edge_idx":edge_idx}
                
                node_idx += 1
                unique_nodes.add(node)
               
        for idx, sub_space in self.sub_spaces_set.items():
            space_mid_point = tuple(sub_space["origin_info"]["mid_point"])
            current_node_idx = self.node_to_idx_dict[space_mid_point]["node_idx"]
            
            connected_mid_point_idxs = {}

            for connected_edges_idx in sub_space["connect_info"]["connected_edges_idx"]:
                p1, p2 = self.idx_to_edge_dict[connected_edges_idx]["edge"]

                connected_edge_type = self.idx_to_edge_dict[connected_edges_idx]["type"]
                
                connected_p1_idx = self.node_to_idx_dict[tuple(p1)]["node_idx"]
                connected_p2_idx = self.node_to_idx_dict[tuple(p2)]["node_idx"]

                connected_p1_type = self.node_to_idx_dict[tuple(p1)]["node_type"]
                connected_p2_type = self.node_to_idx_dict[tuple(p2)]["node_type"]

                if connected_edge_type == "gate_edge":

                    mid_point = tuple(self.idx_to_edge_dict[connected_edges_idx]["mid_point"])
                
                    connected_mid_point_idx = self.node_to_idx_dict[mid_point]["node_idx"]
                    connected_mid_point_idxs[connected_mid_point_idx] = mid_point

                    # add connection between current_node_idx and connected_mid_point
                    self.idx_to_node_dict[current_node_idx]["connected_node_idxs"].add(connected_mid_point_idx)
                    self.node_to_idx_dict[space_mid_point]["connected_node_idxs"].add(connected_mid_point_idx)

                    self.idx_to_node_dict[connected_mid_point_idx]["connected_node_idxs"].add(current_node_idx)
                    self.node_to_idx_dict[mid_point]["connected_node_idxs"].add(current_node_idx)

                    # add connection between current_node_idx and connected_p1
                    if connected_p1_type == "point":

                        if any([check_intersection([space_mid_point, p1], assist_poly) for assist_poly in self.assist_polys]):
                            continue

                        self.idx_to_node_dict[current_node_idx]["connected_node_idxs"].add(connected_p1_idx)
                        self.node_to_idx_dict[space_mid_point]["connected_node_idxs"].add(connected_p1_idx)

                        self.idx_to_node_dict[connected_p1_idx]["connected_node_idxs"].add(current_node_idx)
                        self.node_to_idx_dict[tuple(p1)]["connected_node_idxs"].add(current_node_idx)

                        # add connection between connected_mid_point and connected_p1

                        if any([check_intersection([mid_point, p1], assist_poly) for assist_poly in self.assist_polys]):
                            continue

                        self.idx_to_node_dict[connected_mid_point_idx]["connected_node_idxs"].add(connected_p1_idx)
                        self.node_to_idx_dict[mid_point]["connected_node_idxs"].add(connected_p1_idx)

                        self.idx_to_node_dict[connected_p1_idx]["connected_node_idxs"].add(connected_mid_point_idx)
                        self.node_to_idx_dict[tuple(p1)]["connected_node_idxs"].add(connected_mid_point_idx)

                    # add connection between current_node_idx and connected_p2
                    if connected_p2_type == "point":

                        if any([check_intersection([space_mid_point, p2], assist_poly) for assist_poly in self.assist_polys]):
                            continue

                        self.idx_to_node_dict[current_node_idx]["connected_node_idxs"].add(connected_p2_idx)
                        self.node_to_idx_dict[space_mid_point]["connected_node_idxs"].add(connected_p2_idx)

                        self.idx_to_node_dict[connected_p2_idx]["connected_node_idxs"].add(current_node_idx)
                        self.node_to_idx_dict[tuple(p2)]["connected_node_idxs"].add(current_node_idx)

                        # add connection between connected_mid_point and connected_p2

                        if any([check_intersection([mid_point, p2], assist_poly) for assist_poly in self.assist_polys]):
                            continue

                        self.idx_to_node_dict[connected_mid_point_idx]["connected_node_idxs"].add(connected_p2_idx)
                        self.node_to_idx_dict[mid_point]["connected_node_idxs"].add(connected_p2_idx)

                        self.idx_to_node_dict[connected_p2_idx]["connected_node_idxs"].add(connected_mid_point_idx)
                        self.node_to_idx_dict[tuple(p2)]["connected_node_idxs"].add(connected_mid_point_idx)
                        
    
    
    def generate_ports_v2(self):
        '''
        authors: 
                Yinhang
        Data:
                2024.08.28
        Info:
                generate all ports for each vertex of box skeleton
        '''
        self.calculate_ports()
        
        for ibox, bbox in enumerate(self.valid_bbox):
            if bbox['type'] == "box":
                for vertex in bbox['skeleton']:

                    if self.point_to_idx_dict[tuple(vertex)]["point_type"] == "obs":
                        continue

                    center = np.around(self.valid_bbox[ibox]['center'],2)
                    box_id, nearest_two_points = self.find_box_id_and_vertex(center, self.bbox_valid_ports, vertex)
                    
                    all_ports = []
                    for pt in nearest_two_points:
                        # skip the vertex if it has no ports
                        if len(self.bbox_valid_ports[box_id]['vertex_ports'][tuple(pt)]) == 0:
                            continue
                        ports = self.bbox_valid_ports[box_id]['vertex_ports'][tuple(pt)].reshape(2,2)
                        ports = np.round(ports,2).tolist()
                        all_ports.extend(ports)
                    
                    self.vertices_ports_dict[tuple(vertex)] = {'all_ports': all_ports, 'current_port': None}

    
    def prepare_evader_goals(self, evader, seed):

        evader_goals = []

        pursuer_space_node_idxs = {}
        for pursuer_idx, pursuer_pos in evader.robot_memory_pos.items():
            pursuer_space_node_idx = self.transform_state_to_node_idx(pursuer_pos, whether_space=True)
            pursuer_space_node_idxs[pursuer_idx] = pursuer_space_node_idx

        if self.current_pursuer_space_node_idxs is None or self.current_pursuer_space_node_idxs != pursuer_space_node_idxs:
            self.current_pursuer_space_node_idxs = pursuer_space_node_idxs
            
            distance_to_pursuers = {}
            for node_idx in self.simple_graph.nodes:
                if self.idx_to_node_dict[node_idx]["node_type"] != "sub_space":
                    continue
                
                if len(pursuer_space_node_idxs) == 0:
                    distance_to_pursuers[node_idx] = 1
                
                else:
                    distances = []
                    for pursuer_idx, pursuer_space_node_idx in pursuer_space_node_idxs.items():
                        # distances.append(nx.shortest_path_length(self.simple_graph, source=node_idx, target=pursuer_space_node_idx, weight="weight"))
                        distances.append(self.distance_info_simple_graph[node_idx][pursuer_space_node_idx]["distance"])
                    distance_to_pursuers[node_idx] = (1-NODE_COEFF1)*min(distances) + NODE_COEFF1*sum(distances)/len(distances)

            if EVADER_POLICY == "RETREAT": 
                max_distance = max(distance_to_pursuers.values())
                for node_idx in distance_to_pursuers:
                    distance_to_pursuers[node_idx] /= max_distance

            elif EVADER_POLICY == "SURROUND":
                mean_distance = np.mean(list(distance_to_pursuers.values()))
                # std_distance = np.std(list(distance_to_pursuers.values()))
                for node_idx in distance_to_pursuers:
                    distance_to_pursuers[node_idx] = 1 / (1 + abs(distance_to_pursuers[node_idx] - mean_distance))
            
            self.distance_to_pursuers = distance_to_pursuers

        if EVADER_MODE == "GLOBAL_TRAVEL":
            # 找到betweenness_centrality1中最大的5个space node
            space_node_idxs = [node_idx for node_idx in self.betweenness_centrality.keys() if self.idx_to_node_dict[node_idx]["node_type"] == "sub_space"]

            node_centralities = {node_idx: (1-NODE_COEFF3)*self.betweenness_centrality[node_idx] + NODE_COEFF3*self.distance_to_pursuers[node_idx] for node_idx in space_node_idxs}

            space_node_idxs = sorted(space_node_idxs, key=lambda node_idx: node_centralities[node_idx],reverse=True)

            for node_idx in space_node_idxs:
                if node_idx not in evader_goals and node_idx not in pursuer_space_node_idxs.values():
                    evader_goals.append(node_idx)

                if len(evader_goals) == SAMPLE_NUM:
                    break

        elif EVADER_MODE == "RANDOM_TRAVEL":
            
            np.random.seed(seed)

            # 找到betweenness_centrality1中最大的5个space node
            space_node_idxs = [node_idx for node_idx in self.betweenness_centrality.keys() if self.idx_to_node_dict[node_idx]["node_type"] == "sub_space"]
            
            #为每个node添加一个(0-1)的随机数
            random_centralities = {node_idx: (1-NODE_COEFF2)*self.betweenness_centrality[node_idx] + NODE_COEFF2*np.random.rand() for node_idx in space_node_idxs}

            node_centralities = {node_idx: (1-NODE_COEFF3)*random_centralities[node_idx] + NODE_COEFF3*self.distance_to_pursuers[node_idx] for node_idx in space_node_idxs}
            
            space_node_idxs = sorted(space_node_idxs, key=lambda node_idx: node_centralities[node_idx],reverse=True)

            for node_idx in space_node_idxs:
                if node_idx not in evader_goals and node_idx not in pursuer_space_node_idxs.values():
                    evader_goals.append(node_idx)

                if len(evader_goals) == SAMPLE_NUM:
                    break

        return [np.array(self.idx_to_node_dict[node_idx]["node"]) for node_idx in evader_goals]
    
    
    
    def gen_gates_set(self, evader, pursuers):
        
        pe = evader.state
                   
        self.update_robot_node_dict(pe, pursuers)

        # self.switch_gate_port_by_dist(pe)
        
        if self.graph is None:
            self.build_graph()

            self.distance_info_graph = self.prepare_distance_info(self.graph)

            self.build_simple_graph()
            
            self.distance_info_simple_graph = self.prepare_distance_info(self.simple_graph)
            
            self.calculate_betweenness_centrality(self.simple_graph)

        self.pursuer_DAG = self.build_DAG(self.evader_space_node_idx, list(self.pursuer_space_node_idxs.values()))
        
        critical_nodes = self.filter_critical_nodes(evader, pursuers)

        # deterrence_scores = self.calculate_deterrence_scores(self.graph, node_importance)

        pursuer_idx = list(critical_nodes.keys())
        node_idxs = [critical_nodes[pursuer_idx[i]]['hider_cand'] + critical_nodes[pursuer_idx[i]]['attacker_cand'] for i in range(len(pursuer_idx))]

        node_comb = list(product(*node_idxs))
        
        # 生成笛卡尔积
        pursuer_sub_space_comb_dicts = [dict(zip(pursuer_idx, comb)) for comb in node_comb]

        gates_by_hider_comb = defaultdict(list)

        seq_idx = 0

        # self.evader_paths = nx.single_source_dijkstra_path(self.graph, self.evader_node_idx)
        # self.evader_paths_length = nx.single_source_dijkstra_path_length(self.graph, self.evader_node_idx)
        self.evader_paths_DAG = nx.single_source_dijkstra_path(self.pursuer_DAG, self.evader_space_node_idx)
        # self.evader_paths_simple_graph = nx.single_source_dijkstra_path(self.simple_graph, self.evader_space_node_idx)

        comb_info = {}
        unique_comb = set()
        for comb in pursuer_sub_space_comb_dicts:
            path_lengths = []
            for pursuer_idx, node_id in comb.items():
                path_lengths.append(len(self.distance_info_graph[self.pursuer_node_idxs[pursuer_idx]][node_id]["path"]))
                
            set_of_comb = frozenset(comb.values())
            if set_of_comb not in unique_comb:
                unique_comb.add(set_of_comb)
                comb_info[set_of_comb] = {"comb":comb, "total_path_length":sum(path_lengths)}
                
            else:
                if sum(path_lengths) < comb_info[set_of_comb]["total_path_length"]:
                    comb_info[set_of_comb] = {"comb":comb, "total_path_length":sum(path_lengths)}
            
        filtered_comb = [comb_info[comb]["comb"] for comb in comb_info]

        for comb in filtered_comb:

            valid_comb = True

            # if two space_idx in comb are same, remove it
            # if two space_idx in comb are connected, remove it
            if len(set(comb.values())) != len(comb):
                valid_comb = False
                
            for pursuer_idx, node_id in comb.items():
                connected_node_idxs = self.idx_to_node_dict[node_id]["connected_node_idxs"]
                for connected_node_idx in connected_node_idxs:
                    if connected_node_idx in comb.values():
                        valid_comb = False
                
            # if all comb nodes are point, remove it
            if all(self.idx_to_node_dict[node_id]["node_type"] == "point" for node_id in comb.values()):
                # if any(self.evader_paths_length[node_id] <= DISTANCE_COEFF3 for node_id in comb.values()):
                    valid_comb = False

            # if max_path_length - min_path_length > 1, remove it
            path_lengths = []
            for pursuer_idx, node_id in comb.items():
                # path_lengths.append(critical_nodes[pursuer_idx]['pursuer_path_length'][node_id])
                path_lengths.append(len(self.distance_info_graph[self.pursuer_node_idxs[pursuer_idx]][node_id]["path"]))

            if max(path_lengths) - min(path_lengths) > 3:
                valid_comb = False

            # check if one node in more than one node's neighbor
            hide_node_idxs = set()
            neighbor_of_comb_nodes = []
            for pursuer_idx, node_id in comb.items():
                if node_id in critical_nodes[pursuer_idx]['hider_cand']:
                    hide_node_idxs.add(node_id)
                    # neighbor_nodes = [idx for idx in self.idx_to_node_dict[node_id]["connected_node_idxs"] if self.idx_to_node_dict[idx]["node_type"] == "gate"]
                    # neighbor_of_comb_nodes.append(list(neighbor_nodes) + [node_id])
                    neighbor_of_comb_nodes.append([node_id])

                elif node_id in critical_nodes[pursuer_idx]['attacker_cand']:
                    edge_idx = self.idx_to_node_dict[node_id]["edge_idx"]
                    edge = self.idx_to_edge_dict[edge_idx]["edge"]
                    p1_idx, p2_idx = self.node_to_idx_dict[edge[0]]["node_idx"], self.node_to_idx_dict[edge[1]]["node_idx"]
                    neighbor_of_comb_nodes.append([p1_idx, p2_idx, node_id])
                
            planning_node_idxs = [node_id for net in neighbor_of_comb_nodes for node_id in net]

            for node_id in comb.values():
                if self.idx_to_node_dict[node_id]["node_type"] == "point":
                    evader_path = self.distance_info_graph[self.evader_node_idx][node_id]["path"]
                    # if evader path and neighbor_of_comb_nodes have more than 2 common nodes, remove it
                    if len(set(evader_path) & set(planning_node_idxs)) >= 2:
                        valid_comb = False

            if not valid_comb:
                continue
                    
            gates_info = {"gates_attributes":{}, "closure_polys":[],"deterrance_score":0, "hide_node_idxs":hide_node_idxs,\
                          "pursuer_node_comb:":[], "total_path_length":sum(path_lengths)}
            
            gates_info["pursuer_node_comb"] = comb

            for pursuer_idx, node_id in comb.items():

                neighbor_nodes = self.idx_to_node_dict[node_id]["connected_node_idxs"]

                if node_id in critical_nodes[pursuer_idx]['hider_cand']:

                    vertex = self.idx_to_node_dict[node_id]["node"]

                    gates_info["gates_attributes"][tuple(vertex)] = {"assign_pursuer_idx":pursuer_idx, "gate_type":"hider_cand", "neighbor_nodes":[self.idx_to_node_dict[node_id]["node"] for node_id in neighbor_nodes]}

                elif node_id in critical_nodes[pursuer_idx]['attacker_cand']:
                    
                    edge_idx = self.idx_to_node_dict[node_id]["edge_idx"]

                    gate = self.idx_to_edge_dict[edge_idx]["edge"]

                    gates_info["gates_attributes"][tuple(gate)] = {"assign_pursuer_idx":pursuer_idx,  "gate_type":"attacker_cand", "neighbor_nodes":[self.idx_to_node_dict[node_id]["node"] for node_id in neighbor_nodes]}

            gates_info["deterrance_score"] = self.calculate_nodes_importance(self.evader_node_idx, comb)

            # 将完整的 hide_node_idx (hider_node_idxs) 作为 frozenset 来存储
            gates_by_hider_comb[frozenset(hide_node_idxs)].append(gates_info)

        # 对每个 hider_node_idxs (完整组合) 内部排序并取前 n 种
        # 1. whether_cycle #2. deterrance_score
        n = GATES_CHOOSE_NUM
        gates_set_pre = []
        for hider_comb, gates_list in gates_by_hider_comb.items():
            # check valid_comb
            sorted_gates_list = sorted(gates_list, key=lambda x: x["deterrance_score"], reverse=True)
            gates_set_pre.extend(sorted_gates_list[:n])

        # # sorted by deterrance_score for all
        if GATES_FILTER == "SIMILAR":
            gates_set_pre_sorted = self.sort_gates_by_similarity_and_score(gates_set_pre)
        else:
            gates_set_pre_sorted = sorted(gates_set_pre, key=lambda x: x["deterrance_score"], reverse=True)
        
        # 启发项： 由于动作太多，我们只选择前10个
        gates_set_pre_sorted = gates_set_pre_sorted[:TOTAL_GATES_NUM]

        self.make_gates_set_and_gates_attributes_set(gates_set_pre_sorted, pe)

    
    def update_robot_node_dict(self, pe, pursuers):

        self.evader_node_idx,self.evader_space_node_idx, self.pursuer_node_idxs, self.pursuer_space_node_idxs = None,None, {}, {}

        self.evader_node_idx = self.transform_state_to_node_idx(pe, whether_space=False)

        self.evader_space_node_idx = self.transform_state_to_node_idx(pe, whether_space=True)
        
        for p_idx, pursuer in enumerate(pursuers):
            self.pursuer_node_idxs[p_idx] = self.transform_state_to_node_idx(pursuer.state)

        for p_idx, pursuer in enumerate(pursuers):
            self.pursuer_space_node_idxs[p_idx] = self.transform_state_to_node_idx(pursuer.state, whether_space=True)
    
            
            
    def build_graph(self, show=False):
        # 创建一个有向图
        self.graph = nx.DiGraph()
        
        # 添加节点
        for node_idx in self.idx_to_node_dict.keys():
            self.graph.add_node(node_idx)
        
        # 使用队列来广度优先搜索（BFS）遍历图
        queue = deque([self.evader_node_idx])
        
        visited = set()  # 用来记录已经访问过的节点，避免重复处理
        
        while queue:
            node_idx = queue.popleft()
            visited.add(node_idx)
            
            current_node_type = self.idx_to_node_dict[node_idx]["node_type"]
            
            for connected_node_idx in self.idx_to_node_dict[node_idx]["connected_node_idxs"]:
                
                # 只添加未访问的节点到队列中，防止创建重复边
                if connected_node_idx not in visited:
                    queue.append(connected_node_idx)
                
                if not self.graph.has_edge(node_idx, connected_node_idx):
                    self.graph.add_edge(node_idx, connected_node_idx, weight=np.linalg.norm(np.array(self.idx_to_node_dict[node_idx]["node"]) - np.array(self.idx_to_node_dict[connected_node_idx]["node"])))
         

    
    def build_simple_graph(self, show=False):
        # 创建一个有向图
        self.simple_graph = nx.DiGraph()
        
        # 添加节点
        for node_idx in self.idx_to_node_dict.keys():
                self.simple_graph.add_node(node_idx)
        
        # 使用队列来广度优先搜索（BFS）遍历图
        queue = deque([self.evader_space_node_idx])
        
        visited = set()  # 用来记录已经访问过的节点，避免重复处理
        
        while queue:
            node_idx = queue.popleft()
            visited.add(node_idx)
            
            current_node_type = self.idx_to_node_dict[node_idx]["node_type"]
            
            for connected_node_idx in self.idx_to_node_dict[node_idx]["connected_node_idxs"]:
                # 检查节点类型逻辑，如果需要，这部分代码可以根据需求修改或启用
                
                # if current_node_type == "sub_space" or self.idx_to_node_dict[connected_node_idx]["node_type"] == "sub_space":
                #     continue

                # if current_node_type == "point" and self.idx_to_node_dict[connected_node_idx]["node_type"] == "gate":
                #     continue

                if current_node_type == "point" or current_node_type == "obs":
                    continue

                # if current_node_type == "gate" and self.idx_to_node_dict[connected_node_idx]["node_type"] == "point":
                #     continue
                
                # 只添加未访问的节点到队列中，防止创建重复边
                if connected_node_idx not in visited:
                    queue.append(connected_node_idx)
                
                if not self.simple_graph.has_edge(node_idx, connected_node_idx):
                    self.simple_graph.add_edge(node_idx, connected_node_idx, weight=np.linalg.norm(np.array(self.idx_to_node_dict[node_idx]["node"]) - np.array(self.idx_to_node_dict[connected_node_idx]["node"])))
         
         
    def prepare_distance_info(self, graph):
        
        distance_info = {}

        for node_idx in graph.nodes:
            
            distance_info[node_idx] = {}
            
            reachable_nodes = nx.single_source_dijkstra_path(graph, node_idx)
            reachable_nodes_length = nx.single_source_dijkstra_path_length(graph, node_idx)
            
            for node_jdx in reachable_nodes.keys():
                distance_info[node_idx][node_jdx] = {"distance":reachable_nodes_length[node_jdx], "path":reachable_nodes[node_jdx]}
                
            
        return distance_info
    
    
    def calculate_betweenness_centrality(self, graph, show=False):
        
        # 计算整张图的介数中心性
        self.betweenness_centrality = nx.betweenness_centrality(graph)
        
        min_bc, max_bc = min(self.betweenness_centrality.values()), max(self.betweenness_centrality.values())
        self.betweenness_centrality = {node_idx: (self.betweenness_centrality[node_idx] - min_bc) / (max_bc - min_bc) for node_idx in self.betweenness_centrality}
    
    
    
    def build_DAG(self, start_node_idx, end_node_idxs, show=False):
        
        this_DAG = nx.DiGraph()

        for node_idx in self.graph.nodes:
            this_DAG.add_node(node_idx)
            
        new_end_node_idxs = []
        for end_node_idx in end_node_idxs:
            if self.idx_to_node_dict[end_node_idx]["node_type"] == "point":
                end_node_idxs.extend([idx for idx in self.idx_to_node_dict[end_node_idx]["connected_node_idxs"] if self.idx_to_node_dict[idx]["node_type"] == "gate"])
            
            else:
                new_end_node_idxs.append(end_node_idx)
                
        end_node_idxs = new_end_node_idxs

        queue = deque([start_node_idx])

        visited = set()

        while queue:
            node_idx = queue.popleft()
            visited.add(node_idx)

            current_node_type = self.idx_to_node_dict[node_idx]["node_type"]

            for connected_node_idx in self.idx_to_node_dict[node_idx]["connected_node_idxs"]:

                if current_node_type == "point" or current_node_type == "obs":
                    continue

                if connected_node_idx not in visited and connected_node_idx not in end_node_idxs:
                    queue.append(connected_node_idx)

                if not this_DAG.has_edge(node_idx, connected_node_idx):
                    this_DAG.add_edge(node_idx, connected_node_idx, weight=np.linalg.norm(np.array(self.idx_to_node_dict[node_idx]["node"]) - np.array(self.idx_to_node_dict[connected_node_idx]["node"])))

        return this_DAG
        
        
                
    def filter_critical_nodes(self, evader, pursuers, show=False):
        
        relate_vel_coeff = pursuers[0].max_velocity / evader.velocity

        # evader_paths = nx.single_source_dijkstra_path(self.graph, self.evader_node_idx)
        # evader_paths_length = nx.single_source_dijkstra_path_length(self.graph, self.evader_node_idx)

        critical_nodes = {p_id:{"hider_cand":[],"attacker_cand":[]} for p_id in self.pursuer_node_idxs.keys()}

        # check whether pursuer are overlapping with other pursuers
        whether_overlapping = {p_id: False for p_id in self.pursuer_node_idxs.keys()}
        visited_nodes = {}

        for pursuer_id, pursuer_node_idx in self.pursuer_node_idxs.items():
            if pursuer_node_idx in visited_nodes:
                # 如果当前节点索引已经被其他追捕者占用，标记为重叠
                whether_overlapping[pursuer_id] = True
                whether_overlapping[visited_nodes[pursuer_node_idx]] = True
            else:
                # 记录当前节点索引及其对应的追捕者 ID
                visited_nodes[pursuer_node_idx] = pursuer_id

        for pursuer_id, pursuer_node_idx in self.pursuer_node_idxs.items():
            
            # pursuer_paths = nx.single_source_dijkstra_path(self.graph, pursuer_node_idx)
            # pursuer_paths_length = nx.single_source_dijkstra_path_length(self.graph, pursuer_node_idx)
            
            for node in self.graph.nodes:
                if node not in self.distance_info_graph[pursuer_node_idx].keys():
                    continue
                
                if self.distance_info_graph[self.evader_node_idx][node]["distance"] <= self.distance_info_graph[pursuer_node_idx][node]["distance"]/relate_vel_coeff - DISTANCE_COEFF1*pursuers[0].capture_range:
                    if whether_overlapping[pursuer_id] and len(self.distance_info_graph[pursuer_node_idx][self.evader_node_idx]["path"]) < 3: # 重叠且evader距离追捕者距离较近
                        protect_range = 4
                    else:
                        protect_range = 2
                    
                    if len(self.distance_info_graph[pursuer_node_idx][node]["path"]) > protect_range:
                        continue
                    
                if self.distance_info_graph[pursuer_node_idx][node]["distance"] >= DISTANCE_COEFF4 * self.distance_info_graph[pursuer_node_idx][self.evader_node_idx]["distance"]/relate_vel_coeff:
                    if whether_overlapping[pursuer_id] and len(self.distance_info_graph[pursuer_node_idx][self.evader_node_idx]["path"]) < 3: # 重叠且evader距离追捕者距离较近
                        protect_range = 4
                    else:
                        protect_range = 2

                    if len(self.distance_info_graph[pursuer_node_idx][node]["path"]) > protect_range:
                        continue

                                        
                if self.distance_info_graph[self.evader_node_idx][node]["distance"] >= DISTANCE_COEFF3 and self.idx_to_node_dict[node]["node_type"] == "point": # 藏匿范围超参
                    continue

                if self.distance_info_graph[self.evader_node_idx][node]["distance"] + self.distance_info_graph[pursuer_node_idx][node]["distance"] >= \
                    DISTANCE_COEFF2 * self.distance_info_graph[self.evader_node_idx][self.pursuer_node_idxs[pursuer_id]]["distance"] \
                        and self.idx_to_node_dict[node]["node_type"] == "point": # 搜索范围超参
                    continue
                
                if self.idx_to_node_dict[node]["node_type"] == "point":
                    if self.idx_to_node_dict[node]["node"] not in self.vertices_ports_dict.keys():
                        continue

                    pursuer_pos = pursuers[pursuer_id].state

                    if is_in_sight(evader.state, pursuer_pos, self.obstacles, np.linalg.norm(np.array(evader.state) - np.array(pursuer_pos))):
                        continue

                    memory_nodes = [self.transform_state_to_node_idx(memory_pos) for memory_pos in evader.robot_memory_pos.values()]
                    
                    if any ([len(self.distance_info_graph[memory_node][node]["path"]) < 2 for memory_node in memory_nodes]):
                        continue

                    critical_nodes[pursuer_id]["hider_cand"].append(node)

                elif self.idx_to_node_dict[node]["node_type"] == "gate":
                    critical_nodes[pursuer_id]["attacker_cand"].append(node)

        return critical_nodes
    
    

    def remove_overlap_edges(self, edges):
        
        def step1(edges, skeleton_polys):
            # step 1： 交叉性约束
            edge_dict = {}  
            for edge in edges:

                edge_length = np.linalg.norm(np.array(edge[0]) - np.array(edge[1]))
                line1 = LineString([edge[0], edge[1]])
                overlap = False
                to_remove = []

                if tuple(edge) in skeleton_polys or tuple(edge[::-1]) in skeleton_polys:
                    edge_dict[line1] = (edge, edge_length, "skeleton_edge")
                    
                for line2, (existing_edge, existing_length,edge_type) in edge_dict.items():
                    # 如果两条线段相交，且不是相切的, 且不是相反的
                    if line1.intersects(line2) and (len(set(line1.coords) & set(line2.coords)) != 1):
                        if edge_length < existing_length and edge_type != "skeleton_edge":
                            to_remove.append(line2)    
                        else:
                            overlap = True
                            break
                                
                if not overlap:
                    for line2 in to_remove:
                        del edge_dict[line2]
                    edge_dict[line1] = (edge, edge_length, "gate_edge")
                        
            edges = [existing_edge for existing_edge, _, _ in edge_dict.values()] + [existing_edge[::-1] for existing_edge, _, _ in edge_dict.values()]

            return edges
        
        # step2: 唯一性约束：从点集 A 到点集 B 的映射中，每个点之间的映射应当是一一对应的，或者没有映射。
        # 确认不同ibox之间的唯一连接数目
        
        def step2(edges, skeleton_polys, point_to_idx_dict, idx_to_point_dict, skeleton_polys_dict):
            connect_info = {}
            new_edges = []
            n_points = len(point_to_idx_dict)
            dist_matrix = np.full((n_points, n_points), 1e3)
            closest_dist_matrix = np.full((n_points, n_points), 1e3)
            cost_matrix = np.full((n_points, n_points), 2e2)
            
            for edge in edges:
                p1 = tuple(edge[0])
                p2 = tuple(edge[1])
                
                idx1, skel_idxs1 = point_to_idx_dict[p1]['point_idx'], point_to_idx_dict[p1]['skeleton_idxs']
                idx2, skel_idxs2 = point_to_idx_dict[p2]['point_idx'], point_to_idx_dict[p2]['skeleton_idxs']

                ibox1, ibox2 = point_to_idx_dict[p1]['ibox'], point_to_idx_dict[p2]['ibox']

                edge_length = np.linalg.norm(np.array(p1) - np.array(p2))

                # closest_dist_to_line = min([point_to_line_distance(point, p1, p2) for point in point_to_idx_dict.keys() if point != p1 and point != p2])

                if ibox1 == ibox2:
                    if edge in skeleton_polys or edge[::-1] in skeleton_polys:
                        new_edges.append(edge)

                    else:
                        continue
                
                else:
                    for (skel_idx1, skel_idx2) in product(skel_idxs1, skel_idxs2):
                        if (skel_idx1, skel_idx2) not in connect_info.keys() and (skel_idx2, skel_idx1) not in connect_info.keys():
                            connect_info[(skel_idx1, skel_idx2)] = {"edges":[], "points":{skel_idx1:set(), skel_idx2:set()}, "max_count":0}

                        if (skel_idx1, skel_idx2) in connect_info.keys():
                            connect_info[(skel_idx1, skel_idx2)]["edges"].append(edge)
                            connect_info[(skel_idx1, skel_idx2)]["points"][skel_idx1].add(p1)
                            connect_info[(skel_idx1, skel_idx2)]["points"][skel_idx2].add(p2)
                            connect_info[(skel_idx1, skel_idx2)]["max_count"] = min(len(connect_info[(skel_idx1, skel_idx2)]["points"][skel_idx1]), len(connect_info[(skel_idx1, skel_idx2)]["points"][skel_idx2]))

                        elif (skel_idx2, skel_idx1) in connect_info.keys():
                            connect_info[(skel_idx2, skel_idx1)]["edges"].append(edge)
                            connect_info[(skel_idx2, skel_idx1)]["points"][skel_idx1].add(p1)
                            connect_info[(skel_idx2, skel_idx1)]["points"][skel_idx2].add(p2)
                            connect_info[(skel_idx2, skel_idx1)]["max_count"] = min(len(connect_info[(skel_idx2, skel_idx1)]["points"][skel_idx1]), len(connect_info[(skel_idx2, skel_idx1)]["points"][skel_idx2]))

                dist_matrix[idx1, idx2], dist_matrix[idx2, idx1] = edge_length, edge_length

                # closest_dist_matrix[idx1, idx2], closest_dist_matrix[idx2, idx1] = closest_dist_to_line, closest_dist_to_line

                # cost会考虑edge_length和closest_dist_to_line
                # cost_matrix[idx1, idx2] *= closest_dist_to_line - 0.4
                # cost_matrix[idx2, idx1] *= closest_dist_to_line - 0.4

            import cvxpy as cp

            # 决策变量
            x = cp.Variable((n_points, n_points), boolean=True)

            # 约束条件
            constraints = []
            
            # 点约束：
            # 不同ibox之间的点映射应当是一一对应的，或者没有映射
            # 同一个ibox之间不应当有映射
            for i in range(n_points):
                rel_skel_idxs = idx_to_point_dict[i]['skeleton_idxs']
                for skeleton_idx, skel_info in skeleton_polys_dict.items():
                    if skeleton_idx in rel_skel_idxs:
                        continue
                    point_constraints = []
                    for point_idx in skel_info['point_idxs']:
                        point_constraints.append(x[i, point_idx])
                    constraints.append(cp.sum(point_constraints) <= 1)
                        
            # ibox约束：
            # 对于ibox之间的连接，连接数应该等于max_count
            # for skel1, skel2 in connect_info.keys():
            #     point_idxs1 = skeleton_polys_dict[skel1]['point_idxs']
            #     point_idxs2 = skeleton_polys_dict[skel2]['point_idxs']
            #     ibox_constraints = []
            #     for i in range(len(point_idxs1)):
            #         for j in range(len(point_idxs2)):
            #             ibox_constraints.append(x[point_idxs1[i], point_idxs2[j]])
            #     constraints.append(cp.sum(ibox_constraints) <= connect_info[(skel1, skel2)]["max_count"])
                
            # 对称约束
            for i in range(n_points):
                for j in range(n_points):
                    constraints.append(x[i, j] == x[j, i])
            
            # 目标函数: 最小化总长度
            # z = dist_matrix * x, cost = cost_matrix * (1-x)
            z = cp.sum(cp.multiply(dist_matrix, x))
            q = cp.sum(cp.multiply(closest_dist_matrix, x))
            cost = cp.sum(cp.multiply(cost_matrix, 1-x))
            objective = cp.Minimize(cost + z)
            
            # 构建问题并求解
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.GLPK_MI)
            
            if prob.status != cp.OPTIMAL:
                raise ValueError("Solver did not converge!")
            
            # 提取结果
            x_value = x.value
            for i in range(n_points):
                for j in range(n_points):
                    if x_value[i, j] == 1:
                        new_edges.append([idx_to_point_dict[i]['point'], idx_to_point_dict[j]['point']])

            return new_edges
        
        # 先1后2
        edges = step1(edges, self.skeleton_polys)
        edges = step2(edges, self.skeleton_polys, self.point_to_idx_dict, self.idx_to_point_dict, self.skeleton_polys_dict)
        
        # 先2后1
        # edges = step2(edges, self.skeleton_polys, self.point_to_idx_dict, self.idx_to_point_dict, self.skeleton_polys_dict)
        # edges = step1(edges, self.skeleton_polys)

        return edges
    
    
    def build_closure(self,closure_edges):
        if not closure_edges:
            return []

        # 将边列表转换为字典形式，便于查找连接
        edge_map = {}
        for start, end in closure_edges:
            if start in edge_map:
                edge_map[start].append(end)
            else:
                edge_map[start] = [end]
            if end in edge_map:
                edge_map[end].append(start)
            else:
                edge_map[end] = [start]

        # 从边列表中构建闭包
        closure = []
        visited = set()

        start_vertex = closure_edges[0][0]
        current_vertex = start_vertex

        while len(visited) < len(closure_edges):
            if current_vertex not in visited:
                closure.append(current_vertex)
                visited.add(current_vertex)

            next_vertex = None
            for neighbor in edge_map[current_vertex]:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break

            if next_vertex is None:
                break

            current_vertex = next_vertex

        def polygon_area(vertices):
            n = len(vertices)
            area = 0
            for i in range(n):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return area / 2

        # 判断顺序并反转为逆时针
        if polygon_area(closure) > 0:
            closure = closure[::-1]

        return closure
    
    
    def calculate_ports(self, robot_radius = 0.1):
        '''
        generate ports modified by Junfeng
        1. each vertex for each obstacle has two ports
        '''
        # for ibox in range(0, self.obstacles.shape[2]): # self.obstacles: list[np.array shape(4,2) or shape(6,2)]
        # for i, vertex in enumerate(self.obstacles[:,:,ibox]):
        self.bbox_valid_ports = {}
        for ibox in range(len(self.obstacles)):
            self.bbox_valid_ports[ibox] = {'vertex_ports': {}, 'center': None}
            for i, vertex in enumerate(self.obstacles[ibox].tolist()):
                self.bbox_valid_ports[ibox]['center'] = np.mean(self.obstacles[ibox], axis=0)
                if not is_in_boundary(vertex, self.boundary):
                    continue

                else:
                    # 从self.obstacles[:,:, ibox]找到vertex的邻居顶点
                    pre_ind = (i - 1) % self.obstacles[ibox].shape[0]
                    next_ind = (i + 1) % self.obstacles[ibox].shape[0]
                    
                    # obtain neiborhood vertex
                    # pre_vertex = self.obstacles[pre_ind, :, ibox]
                    # next_vertex = self.obstacles[next_ind, :, ibox]
                    
                    pre_vertex = self.obstacles[ibox][pre_ind,:]
                    next_vertex = self.obstacles[ibox][next_ind,:]
                    
                    # calculate the vector from vertex to pre_vertex and next_vertex
                    vector_pre =  pre_vertex - vertex
                    vector_next = next_vertex - vertex
                    
                    # extend a minor vector to generate the port for deleting vertex
                    minor_pre_vector = vertex + vector_pre / np.linalg.norm(vector_pre) * 0.1
                    minor_pre_vector -= vector_next/ np.linalg.norm(vector_next) * 0.1
                    minor_next_vector = vertex + vector_next / np.linalg.norm(vector_next) * 0.1
                    minor_next_vector -= vector_pre / np.linalg.norm(vector_pre) * 0.1
                    if is_in_obstacle(minor_pre_vector, self.obstacles) or is_in_obstacle(minor_next_vector, self.obstacles):
                        # continue
                        self.bbox_valid_ports[ibox]['vertex_ports'][tuple(vertex)] = []

                    else:
                        port1 = vertex + vector_pre / np.linalg.norm(vector_pre) * robot_radius * 2
                        port1 -= vector_next / np.linalg.norm(vector_next) * robot_radius * 2
                        port2 = vertex + vector_next / np.linalg.norm(vector_next) * robot_radius * 2
                        port2 -= vector_pre / np.linalg.norm(vector_pre) * robot_radius * 2

                        self.ports_vertex[tuple(vertex)] = np.array([np.reshape(port1, (-1, 1)), np.reshape(port2, (-1, 1))])
                        self.bbox_valid_ports[ibox]['vertex_ports'][tuple(vertex)] = np.array([np.reshape(port1, (-1, 1)), np.reshape(port2, (-1, 1))])

                
    def find_box_id_and_vertex(self, center, bbox_valid_ports, vertex):
        bbox_id = None
        for ibox, info in bbox_valid_ports.items():
            if np.array_equal(center, np.round(info['center'],2)):
                bbox_id = ibox
                break
        vertex_list = list(bbox_valid_ports[bbox_id]['vertex_ports'].keys())
        
        # 计算vertex_list中每个点到vertex的距离
        distances = [np.linalg.norm(np.array(point) - np.array(vertex)) for point in vertex_list]

        # 创建一个包含点和对应距离的元组的列表
        points_and_distances = list(zip(vertex_list, distances))

        # 根据距离对点进行排序
        points_and_distances.sort(key=lambda x: x[1])

        # 获取距离vertex最近的两个点
        nearest_two_points = [point for point, distance in points_and_distances[:2]] #[point1, point2]

        return bbox_id, nearest_two_points

        
    def transform_state_to_node_idx(self, state, whether_space=False):

        def is_point_in_or_touching_polygon(point, poly):
            return Polygon(poly).contains(Point(point)) or Polygon(poly).touches(Point(point))
        
        for sub_space_idx, sub_space in self.sub_spaces_set.items():
            if is_point_in_or_touching_polygon(state, sub_space['origin_info']['vertex']):
                current_sub_space_idx = sub_space_idx
            
        space_mid_point = self.sub_spaces_set[current_sub_space_idx]['origin_info']['mid_point']

        current_space_node_idx = self.node_to_idx_dict[tuple(space_mid_point)]['node_idx']

        if not whether_space:
            neighbors = [idx for idx in self.idx_to_node_dict[current_space_node_idx]["connected_node_idxs"]] \
                        + [current_space_node_idx]
        else:
            neighbors = [current_space_node_idx]

        node_dict = {idx: self.idx_to_node_dict[idx]["node"] for idx in list(neighbors)}
            
        def find_nearest_node_idx(node_dict, state):

            min_distance = np.inf
            nearest_idx = None

            for idx, node in node_dict.items():
                distance = np.linalg.norm(np.array(node) - np.array(state))
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = idx

            return nearest_idx
        
        nearest_node_idx = find_nearest_node_idx(node_dict, state)

        return nearest_node_idx
    
    
            
    def calculate_nodes_importance(self, start_node, comb_nodes_dict):

        # 计算包含comb_nodes_dict节点的路径/所有路径的比例

        upper, lower = 0, len(self.distance_info_graph[start_node])
        pursuer_num = len(comb_nodes_dict)

        if pursuer_num > 1:
            weight = np.log(1+1/lower)/np.log(pursuer_num)
        else:
            weight = 1

        protected_nodes = {p_id:[] for p_id in comb_nodes_dict.keys()}

        for p_id, node_id in comb_nodes_dict.items():
            protected_nodes[p_id].append(node_id)
            
            if self.idx_to_node_dict[node_id]["node_type"] == "point":
                protected_nodes[p_id].extend([idx for idx in self.idx_to_node_dict[node_id]["connected_node_idxs"] if self.idx_to_node_dict[idx]["node_type"] == "gate"])
            
            elif self.idx_to_node_dict[node_id]["node_type"] == "gate":
                protected_nodes[p_id].extend([idx for idx in self.idx_to_node_dict[node_id]["connected_node_idxs"] if self.idx_to_node_dict[idx]["node_type"] == "point"])

            # exclude start_node
            # protected_nodes[p_id] = [node for node in protected_nodes[p_id] if (node != start_node and node != node_id)]

        # 提前计算路径集合和保护节点集合
        path_sets = {
            end_node: set(self.distance_info_graph[start_node][end_node]["path"])
            for end_node in self.distance_info_graph[start_node].keys()
        }
        protected_nodes_sets = {p_id: set(nodes) for p_id, nodes in protected_nodes.items()}

        # 计算 upper
        upper = 0
        for end_node in path_sets.keys():
            count = 0
            for p_id, nodes_set in protected_nodes_sets.items():
                if path_sets[end_node] & nodes_set:
                    count += 1
                    if count == pursuer_num:  # 提前终止
                        break
            upper += count**weight / pursuer_num**weight

        nodes_importance = upper / lower

        if any([self.idx_to_node_dict[node_id]["node_type"] == "point" for node_id in comb_nodes_dict.values()]):
            if nodes_importance <= 0.7:  # 包含point node组合需要满足的条件
                nodes_importance = 0 
        else:
            nodes_importance = 0.5 * nodes_importance
            
        return nodes_importance
    
        
    def make_gates_set_and_gates_attributes_set(self, gates_set_pre, pe):

        gates_set_pre = gates_set_pre

        self.gates_set = {idx:{"gates_attributes":{},"pursuer_node_comb":[],"deterrance_score":0} for idx in range(len(gates_set_pre))}
        
        for idx, gates_info in enumerate(gates_set_pre):

            for gate, gate_attributes in gates_info['gates_attributes'].items():
                if gate_attributes['gate_type'] == "attacker_cand":
                    vertex1, vertex2 = np.array(gate[0]), np.array(gate[1])
                    ibox1, ibox2 = self.point_to_idx_dict[tuple(vertex1)]['ibox'], self.point_to_idx_dict[tuple(vertex2)]['ibox']
                    type1, type2 = self.valid_bbox[ibox1]['type'], self.valid_bbox[ibox2]['type']
                    center1, center2 = self.valid_bbox[ibox1]['center'], self.valid_bbox[ibox2]['center']

                    gate_attributes = {'data':tuple(gate), 'type':(type1, type2), 'center': (center1, center2), 'assign_pursuer_idx':gate_attributes['assign_pursuer_idx']\
                        , "gate_type":gate_attributes["gate_type"], "neighbor_nodes":gate_attributes["neighbor_nodes"]}
                                        
                    new_gate_attributes = self.find_vertices_ports_of_gate(gate, gate_attributes, pe)

                else:
                    new_gate_attributes = gate_attributes

                self.gates_set[idx]["gates_attributes"][gate] = new_gate_attributes

                self.gates_set[idx]["pursuer_node_comb"] = gates_info["pursuer_node_comb"]

                self.gates_set[idx]["deterrance_score"] = gates_info["deterrance_score"]

    
        
    def find_vertices_ports_of_gate(self, gate, gate_attributes, pe, robot_radius = 0.1, show = False):
        '''
           authors: 
                    Modified by Yinhang and Junfeng
            Date: 
                    2024.07.11
            Info:
                    get all vertexs inside the circle generated by the gate center and gate radius
        '''
        all_valid_vertices = []
        vertex_ports = {}
        
        for vertex in gate:

            if tuple(vertex) in self.vertices_ports_dict.keys():
                all_valid_vertices.append(vertex)
                all_ports = self.vertices_ports_dict[tuple(vertex)]['all_ports']
                vertex_ports[tuple(vertex)] = list(all_ports)

        gate_attributes['vertex_ports'] = vertex_ports
        gate_attributes['all_vertices'] = [tuple(vertex) for vertex in all_valid_vertices]

        if show:
            # plot skeletons, gate and ports
            plt.figure()
            plt.title("generate ports")

            for ibox in range(len(self.valid_bbox)):
                skeleton = self.valid_bbox[ibox]['skeleton']
                plt.plot(skeleton[:,0], skeleton[:,1], 'r-')

            for vertex in all_valid_vertices:
                plt.scatter(vertex[0], vertex[1], color='blue', marker='o')

            plt.plot([gate[0][0], gate[1][0]], [gate[0][1], gate[1][1]], 'b-', zorder=10)

            for vertex_ports in vertex_ports.values():
                for port in vertex_ports:
                    plt.scatter(port[0], port[1], color='pink', marker='*')

            plt.scatter(pe[0], pe[1], color='black', marker='*', zorder=40)

            # plot circles
            circle = plt.Circle(gate_center, gate_radius, color='green', fill=False)
            plt.gca().add_artist(circle)

            plt.xlim(self.boundary[0][0], self.boundary[0][1])
            plt.ylim(self.boundary[1][0], self.boundary[1][1])

            plt.pause(10)
            # plt.savefig('ports.png')

        # return valid_ports, gate_attributes
        return gate_attributes

              