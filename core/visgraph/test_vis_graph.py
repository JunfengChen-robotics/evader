import os, sys
path = os.path.abspath(os.path.join(os.getcwd(), "."))
DC_path = os.path.join(path)
sys.path.append(DC_path)
from vis_graph import VisGraph
from graph import Point, array_to_points
import matplotlib.pyplot as plt
import numpy as np
import time
from NewCode.src.tools.utils.utils import poses2polygons, is_in_obstacle, is_in_boundary, is_dist_boundary, generate_points, is_point_in_polygon, inflate_line, points_in_polygon

# def remove_path_vertices_from_polygons(path, polygons):
#     vertices_to_remove = set((vertex.x, vertex.y) for vertex in path[1:-1])
    
#     new_polygons = []
#     for polygon in polygons:
#         # Create a new polygon excluding the vertices to remove
#         new_polygon = [vertex for vertex in polygon if (vertex.x, vertex.y) not in vertices_to_remove]
#         # Only add the polygon if it still has vertices after removal
#         if new_polygon:
#             new_polygons.append(new_polygon)
    
#     return new_polygons

start_point = Point(0.2, 0.5)  
end_point = Point(9.8, 9.5)  
best_gate = np.array([[4.0, 1.7], [6.5, 4.3]])
blocked_edge = (array_to_points(best_gate[0]), array_to_points(best_gate[1]))
# x0, y0 = best_gate[0,:]
# x1, y1 = best_gate[1,:]
# best_gate_poly = inflate_line(x0, y0, x1, y1, line_width= 0)
# best_gate_poly = [[4,1.5], [6.5, 4.0],[6.5, 4.5], [4, 2.0] ]
polygons_coords =[[[0.0, 1.5], [4.0, 1.5], [4.0, 2.0], [0, 2.0]], [[5.5, 0], [6.0, 0], [6, 2.8], [5.5, 2.8]],
 [[6.5, 4.0], [9, 4.0], [9.0, 4.5], [6.5, 4.5]], 
 [[2.0, 2.8], [2.5,2.8], [2.5, 6.5], [2.0, 6.5]],
  [[4.5, 6.0], [7.5, 6.0], [7.5, 6.5], [4.5, 6.5]],
  [[5.5, 8.5], [6, 8.5], [6.0, 10], [5.5, 10]], 
  [[8.5, 7], [9, 7], [9, 10], [8.5,10]],
   [[1.2, 7.5], [3.5, 7.5], [3.5, 9], [3.0, 9.0], [3.0, 8.0], [1.2, 8.0]]]

boundary= [[0, 10], [0, 10.1]]
# polygons_coords =[[[3.0, 3.5], [3.3, 3.5], [3.3, 5.0], [3.0, 5.0]], [[3.5, 1.5], [5.0, 1.5],
#     [5.0, 1.8], [3.5, 1.8]],[[3.0,2.5],[4.0,2.5],[4.0,2.8],[3.0,2.8]], [[0.0, 0.5], [1.3, 0.5], [1.3, 0.8], [0.0, 0.8]], [[
#       2.2, 0.0], [2.5, 0.0], [2.5, 1.5], [2.2, 1.5]], [[0.7, 1.5], [1.0, 1.5], [1.0,
#       3.2], [2.2, 3.2], [2.2, 3.5], [0.7, 3.5]]]
# polygons_coords.append(best_gate_poly) # method2

polygons = []
for coords in polygons_coords:
    polygon = [Point(x, y) for x, y in coords]
    polygons.append(polygon)
    

workers = 1 
graph = VisGraph()
graph.build(polygons, workers=workers, boundary=boundary)
print('Finished building visibility graph')

# for edge in graph.visgraph.edges:
#     plt.plot([edge.p1.x, edge.p2.x], [edge.p1.y, edge.p2.y], 'r-')
        
start_time = time.time()
# path = graph.shortest_path(start_point, end_point, solver = "dijkstra")
path = graph.shortest_path(start_point, end_point, blocked_edge=blocked_edge, solver = "astar")# method1
# path = graph.shortest_path(start_point, end_point, blocked_edge=None, solver = "astar")  # method2
# path = graph.shortest_path(start_point, end_point, blocked_edge=blocked_edge, solver = "dijkstra")  # method2

end_time = time.time()
print('Time taken: ', time.time() - start_time)

fig, ax = plt.subplots()
for polygon in polygons:
    x_coords = [point.x for point in polygon] + [polygon[0].x]  
    y_coords = [point.y for point in polygon] + [polygon[0].y]  
    ax.plot(x_coords, y_coords, linestyle='-', color='blue', linewidth=0.5, marker='')


x_coords = [point.x for point in path]
y_coords = [point.y for point in path]

ax.plot(best_gate[:, 0], best_gate[:, 1], linestyle='--', color='k', linewidth=2, marker='')
ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=5, label='Path')
ax.plot(x_coords[0], y_coords[0], marker='o', color='red', markersize=10, label='Start')
ax.plot(x_coords[-1], y_coords[-1], marker='o', color='green', markersize=10, label='End')

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])


plt.legend()
plt.show()
print('Done')

