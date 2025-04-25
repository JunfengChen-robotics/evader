"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
import numpy as np
import os,sys
path = os.path.abspath(os.path.join(os.getcwd(), "."))
DC_path = os.path.join(path)
sys.path.append(DC_path)
import math

import matplotlib.pyplot as plt

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
        
        # self.add_cost_set = None


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
        # plt.matshow(self.occ_map_obs)
        # plt.colorbar()
        # print("obstacle_map generated")
        # plt.show()

    def gcost_add(self, current_node, update_node, move_cost):
        # start_time = time.time()  
        
        update_x, update_y = update_node
        
        if update_x < 0 or update_x >= self.occ_map_obs.shape[0] or update_y < 0 or update_y >= self.occ_map_obs.shape[1]:
            return float('inf') 
    
        map_cost = self.occ_map_obs[update_x][update_y] 

        cost = current_node.cost + move_cost + map_cost

        # end_time = time.time()  
        # print(f"Execution time: {end_time - start_time} seconds")
        return cost
    
    @staticmethod
    def compute_cost_heuristic(args):
                o, node_data, goal_node, calc_heuristic_func = args
                return o, node_data.cost + calc_heuristic_func(goal_node, node_data)
    
    
    def planning(self, sx, sy, gx, gy, mode= None, robot_name=None):
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

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print(f"Open set is empty.. for {robot_name}, \n Mode is: {mode}, \n Current position: {sx, sy}, \n Goal position: {gx, gy}")
                # raise ValueError("Cannot find path")
                raise ValueError("Cannot find path")
                break
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
                    # self.ax.savefig(f"astar.png")
                    
                    

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
                                               self.motion[i][2]),
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
        # if self.occ_map_obs[node.x][node.y] == 0:
        if self.occ_map_obs[node.x][node.y] == np.inf:
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

