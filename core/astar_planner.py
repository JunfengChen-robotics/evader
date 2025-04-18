"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from .utils import * 

show_animation = False


class AstarPlanner():
    def __init__(self, world):

        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        xy_res: grid xy_res [m]
        rr: robot radius[m]
        """
        self.world = world
        self.xy_res = world.grid_map.xy_res
        self.boundary = world.grid_map.boundary
        
        self.nx = world.grid_map.nx
        self.ny = world.grid_map.ny
        
        self.rr = 0.1
        self.motion = self.get_motion_model()
        


    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)
    
    def process_map(self, occ_map):
        self.occ_map_obs = occ_map

    def gcost_add(self, fixed_node, update_node_coordinate, i, added_obs_soft=None, add_cost=1e3):
        gc = self.motion[i][2]
        update_x, update_y = update_node_coordinate

        if isinstance(added_obs_soft, set):
            added_obs_soft = np.array(list(added_obs_soft))
        
        if added_obs_soft is not None and len(added_obs_soft) > 0:
            # if (update_x, update_y) in added_obs_soft:
            if np.any((added_obs_soft[:, 0] == update_x) & (added_obs_soft[:, 1] == update_y)):    
                if show_animation:
                    px = self.calc_grid_position(update_x, self.boundary[0, 0])
                    py = self.calc_grid_position(update_y, self.boundary[1, 0])
                    plt.plot(px, py, "xr")
                return fixed_node.cost + gc + add_cost

        return fixed_node.cost + gc
    
    
    def planning(self, sx, sy, gx, gy,added_obs_soft=None):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: end x position [m]
            gy: end y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        
        start_node = self.Node(self.calc_xy_index(sx, self.boundary[0,0]),
                               self.calc_xy_index(sy, self.boundary[1,0]), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.boundary[0,0]),
                              self.calc_xy_index(gy, self.boundary[1,0]), 0.0, -1)

        def change_node_state(goal_node):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1), (2,0), (-2,0), (0, 2), (0, -2), (2, 2), (-2, -2), (2, -2), (-2, 2)]
            for dx, dy in directions:
                temp_goal_x, temp_goal_y = goal_node.x, goal_node.y
                while is_in_obstacle(np.array([goal_node.x / 10, goal_node.y / 10]), self.world.obstacles) or is_at_obstacle_vertex_or_edge(np.array([goal_node.x /10, goal_node.y /10]), self.world.obstacles):
                    goal_node.x += dx
                    goal_node.y += dy
                    if is_at_obstacle_vertex_or_edge(np.array([goal_node.x /10, goal_node.y /10]), self.world.obstacles):
                        goal_node.x, goal_node.y = temp_goal_x, temp_goal_y
                        break
                    elif is_in_obstacle(np.array([goal_node.x / 10, goal_node.y / 10]), self.world.obstacles):
                        goal_node.x, goal_node.y = temp_goal_x, temp_goal_y
                        break
                    else:
                        break
            return goal_node
        
        start_node = change_node_state(start_node)
        goal_node = change_node_state(goal_node)
        
        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty.. for astar_planner_single")
                # raise ValueError("Cannot find path")
                break
            
            # start_time = time.time()
            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            # end_time = time.time()
            # print(f"Execution time: {end_time - start_time} seconds")
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.boundary[0,0]),
                         self.calc_grid_position(current.y, self.boundary[1,0]), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Find end")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                #  current.cost + self.motion[i][2],
                                self.gcost_add(current, 
                                               (current.x + self.motion[i][0], current.y + self.motion[i][1]),
                                               i,
                                               added_obs_soft=added_obs_soft),
                                 c_id)
                               
                
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        
        self.path = np.array([rx,ry]).T
        # self.path = np.array([rx, ry]).T[::-1]
        return self.path

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.boundary[0,0])], [
            self.calc_grid_position(goal_node.y, self.boundary[1,0])]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.boundary[0,0]))
            ry.append(self.calc_grid_position(n.y, self.boundary[1,0]))
            parent_index = n.parent_index

        return rx, ry
    
    @staticmethod #Euclidean
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        dx = n1.x - n2.x
        dy = n1.y - n2.y
        return w * math.sqrt(dx * dx + dy * dy)
    

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.xy_res + min_position + self.xy_res * 0.5
        # pos = index * self.xy_res + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return int(np.floor((position - min_pos) / self.xy_res))

    def calc_grid_index(self, node):
        return (node.y - self.boundary[1,0]) * self.nx + (node.x - self.boundary[0,0])

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.boundary[0,0])
        py = self.calc_grid_position(node.y, self.boundary[1,0])

        if px < self.boundary[0,0]:
            return False
        elif py < self.boundary[1,0]:
            return False
        elif px >= self.boundary[0,1]:
            return False
        elif py >= self.boundary[1,1]:
            return False

        # collision check
        if self.occ_map_obs[node.x][node.y]:
            return False

        return True  
        
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


