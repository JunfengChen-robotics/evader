import numpy as np
from matplotlib.path import Path
from shapely.geometry import LineString, Polygon, Point


def find_skeleton(coords):
    coords = np.array(coords)
    if len(coords) == 4:
        return find_rectangle_skeleton(coords)
    elif len(coords) == 6:
        return find_l_shape_skeleton(coords)
    elif len(coords) == 8:
        return find_s_shape_skeleton(coords)
    elif len(coords) == 2:
        return np.array([coords[0], coords[1]])
    elif len(coords) == 1:
        return np.array([coords[0]])

    else:
        raise ValueError("Unsupported shape with given coordinates")
    
    
def find_rectangle_skeleton(coords):
    coords = np.array(coords)
    
    # 计算每条边的长度
    lengths = [np.linalg.norm(coords[i] - coords[(i + 1) % 4]) for i in range(4)]
    
    # 找到短边的索引
    short_edges = sorted(range(4), key=lambda i: lengths[i])[:2]
    
    # 计算短边的中点
    mid_point1 = (coords[short_edges[0]] + coords[(short_edges[0] + 1) % 4]) / 2
    mid_point2 = (coords[short_edges[1]] + coords[(short_edges[1] + 1) % 4]) / 2
    
    interpolated_skeleton = interpolate_skeleton([mid_point1, mid_point2])

    return interpolated_skeleton



def find_l_shape_skeleton(coords):
    coords = np.array(coords)
    skeleton = None
    
    def is_parallel_and_same_direction(vec1, vec2):
        return np.dot(vec1, vec2) > 0 and np.cross(vec1, vec2) == 0

    for i in range(6):
        coords_set_one = [coords[i], coords[(i - 1) % 6], coords[(i - 2) % 6]]
        coords_set_two = [coords[(i+1)%6], coords[(i + 2) % 6], coords[(i + 3) % 6]]

        is_parallel = True

        for j in range(2):
            vec1 = coords_set_one[j+1] - coords_set_one[j]
            vec2 = coords_set_two[j+1] - coords_set_two[j]

            # 保留两位小数
            vec1, vec2 = np.round(vec1, 2), np.round(vec2, 2)

            if not is_parallel_and_same_direction(vec1, vec2):
                is_parallel = False
                break

        if is_parallel:
            skeleton = [np.array((coords_set_one[i] + coords_set_two[i]) / 2) for i in range(3)]

            interpolated_skeleton = interpolate_skeleton(skeleton)

            # 保留两位小数
            return interpolated_skeleton
        
        
        
def find_s_shape_skeleton(coords):
    coords = [np.round(coord, 2) for coord in coords]
    skeleton = None
    
    def is_parallel_and_same_direction(vec1, vec2):
        return np.dot(vec1, vec2) > 0 and np.cross(vec1, vec2) == 0 and max(np.linalg.norm(vec1), np.linalg.norm(vec2))/min(np.linalg.norm(vec1), np.linalg.norm(vec2)) < 2

    vertex_num = len(coords)

    for i in range(vertex_num):
        coords_set_one = [coords[i], coords[(i - 1) % vertex_num], coords[(i - 2) % vertex_num], coords[(i - 3) % vertex_num]]
        coords_set_two = [coords[(i+1)%vertex_num], coords[(i + 2) % vertex_num], coords[(i + 3) % vertex_num], coords[(i + 4) % vertex_num]]

        is_parallel = True

        for j in range(3):
            vec1 = coords_set_one[j+1] - coords_set_one[j]
            vec2 = coords_set_two[j+1] - coords_set_two[j]

            vec1, vec2 = np.round(vec1, 2), np.round(vec2, 2)

            if not is_parallel_and_same_direction(vec1, vec2):
                is_parallel = False
                break

        if is_parallel:
            skeleton = [np.array((coords_set_one[i] + coords_set_two[i]) / 2) for i in range(4)]

            # 保留两位小数
            skeleton = [np.round(point, 2) for point in skeleton]

            valid = True
            # for point in skeleton:
            #     if not (Polygon(coords).contains(Point(point)) or Polygon(coords).intersects(Point(point))):
            #         valid = False
            #         break

            if valid:

                interpolated_skeleton = interpolate_skeleton(skeleton)
                return interpolated_skeleton
            

def interpolate_skeleton(skeleton, hyperparameter=2):
    skeleton = np.array(skeleton)
    interpolated_points = []

    for i in range(len(skeleton) - 1):
        # 获取当前点和下一点
        start = skeleton[i]
        end = skeleton[i + 1]
        
        # 计算长度(取上界) 2是超参数
        length = int(np.ceil(np.linalg.norm(end - start)/hyperparameter))
        
        # 计算每个点的位置
        for j in range(length):
            interpolated_points.append(start + (end - start) * (j / length))

    # 添加最后一个点（终点）
    interpolated_points.append(skeleton[-1])

    # 将所有点取两位小数
    return [np.round(point, 2) for point in interpolated_points]



def project_point_to_line(point, line):
    point = np.array(point)
    line = np.array(line)
    
    line_vector = line[1] - line[0]
    
    point_vec = point - line[0]
    
    line_len_sq = np.dot(line_vector, line_vector)
    
    factor = np.dot(point_vec, line_vector) / line_len_sq
    
    projection = line[0] + factor * line_vector
    
    return [round(num,3) for num in projection.tolist()]


def is_in_obstacle(ports, obstacles):
    """
    Check if each of the ports is inside any of the obstacles.
    `ports` is a list or array of points, where each point is a 2-element list or array representing the coordinates.
    `obstacles` is a list of obstacles, where each obstacle is represented by a list of its vertex coordinates.
    Returns a boolean array where each element corresponds to whether the respective point is in any obstacle.
    """
    # ports = np.array(ports)
    ports = np.atleast_2d(ports)
    result = np.zeros(len(ports), dtype=bool)
    
    for obstacle in obstacles:
        polygon_path = Path(obstacle)
        result = result | polygon_path.contains_points(ports)
        
    return result


def is_at_obstacle_vertex_or_edge(pos, obstacles):
    """
    Check if the position is at any of the obstacle vertices or on the edges.
    
    :param pos: The position to check, given as a tuple (x, y).
    :param obstacles: A list of obstacles, where each obstacle is represented by a list of its vertex coordinates.
    :return: True if the position is at any of the obstacle vertices or on the edges, False otherwise.
    """
    pos = np.array(pos)
    for obstacle in obstacles:
        for i in range(len(obstacle)):
            vertex = np.array(obstacle[i])
            if np.array_equal(pos, vertex):
                return True
            next_vertex = np.array(obstacle[(i + 1) % len(obstacle)])
            if is_point_on_line_segment(pos, vertex, next_vertex):
                return True
    return False

        
def is_point_on_line_segment(point, line_start, line_end):
    """
    Check if a point is on a line segment.
    
    :param point: The point to check, given as a numpy array [x, y].
    :param line_start: The start point of the line segment, given as a numpy array [x, y].
    :param line_end: The end point of the line segment, given as a numpy array [x, y].
    :return: True if the point is on the line segment, False otherwise.
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.array_equal(point, line_start)
    line_unit_vec = line_vec / line_len
    projection_length = np.dot(point_vec, line_unit_vec)
    projection_point = line_start + projection_length * line_unit_vec
    return np.allclose(point, projection_point) and 0 <= projection_length <= line_len



def is_intersect_boundary(gate, boundary):
    """
    Check if the gate is insect the boundary.
    """

    insect = False
    for point in gate:
        if point[0] == boundary[0][0] or point[0] == boundary[0][1] or point[1] == boundary[1][0] or point[1] == boundary[1][1]:
            insect = True
            break

    return insect


def is_almost_equal(a, b, tolerance=0.01):
        return abs(a - b) < tolerance
    
    
def gate_near_boundary(gate, boundary):
    """
    Check if the gate is on the boundary. 

    two points of the gate are both on the boundary.
    """

    near_boundary = [False, False]
    for i, point in enumerate(gate):
        if point[0] == boundary[0][0] or point[0] == boundary[0][1] or point[1] == boundary[1][0] or point[1] == boundary[1][1]:
            near_boundary[i] = True

    if near_boundary[0] and near_boundary[1]:
        return True
    else:
        return False
    
    
def is_in_sight(pos1, pos2, obs_info_local_,scout_range):
    """
    判断端口是否在 Evader 的视线范围内。判断evader 是否在pursuer 视线范围内。或pursuer是否在evader视线范围内。
    即端口到 Evader 之间的距离是否小于感知evader的范围，以及连线是否没有障碍物遮挡。
    """

    if np.linalg.norm(np.array(pos1) - np.array(pos2)) > scout_range:
        return False

    for obs_verts in obs_info_local_:
        if obs_verts.shape[0] == 2:
            obs_verts = obs_verts.T
        if line_intersects_obstacle(pos1, pos2, obs_verts):
            return False

    return True


def line_intersects_obstacle(pos1, pos2, obstacle):
    """
    检查由 pos2 和 pos1 定义的线段是否与障碍物相交。
    
    :param pos2: 线段的一个端点，格式为 (x, y)
    :param pos1: 线段的另一个端点，格式为 (x, y)
    :param obstacle: 障碍物的顶点列表，格式为 [(x1, y1), (x2, y2), ...]
    :return: 如果线段与障碍物相交，则返回 True，否则返回 False
    """
    line = LineString([pos2, pos1])
    poly = Polygon(obstacle)
    
    return line.intersects(poly)



def is_in_boundary(pos, boundary):
    """
    Check if the port is inside the boundary.

    `port` is a 2-element list or array representing the coordinates of the port.
    `boundary` is a list of the boundary's vertex coordinates.
    """
    inside = True
    for i in range(0, 2):
        if pos[i] <= boundary[i, 0] or pos[i] >= boundary[i, 1]:
            inside = False

    return inside