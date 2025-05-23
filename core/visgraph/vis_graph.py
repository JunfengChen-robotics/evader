"""
The MIT License (MIT)

Copyright (c) 2016 Christian August Reksten-Monsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from timeit import default_timer
from sys import stdout, version_info
from multiprocessing import Pool
from warnings import warn
from .graph import Graph, Edge
from .shortest_path import shortest_path_glb
from .visible_vertices import closest_point, point_in_polygon,visible_vertices

PYTHON3 = version_info[0] == 3
if PYTHON3:
    xrange = range
    import pickle
else:
    import cPickle as pickle


class VisGraph(object):

    def __init__(self):
        self.graph = None
        self.visgraph = None
        self.boundary = None

    def load(self, filename):
        """Load obstacle graph and visibility graph. """
        with open(filename, 'rb') as load:
            self.graph, self.visgraph = pickle.load(load)

    def save(self, filename):
        """Save obstacle graph and visibility graph. """
        with open(filename, 'wb') as output:
            pickle.dump((self.graph, self.visgraph), output, -1)

    def edge_in_boundary(self, edge):
        if edge.p1.x == edge.p2.x and (edge.p1.x == self.boundary[0][0] or edge.p1.x == self.boundary[0][1]):
            return True
        elif edge.p1.y == edge.p2.y and (edge.p1.y == self.boundary[1][0] or edge.p1.y == self.boundary[1][1]):
            return True
        return False

    def build(self, input, workers=1, boundary = None, status=True):
        """Build visibility graph based on a list of polygons.

        The input must be a list of polygons, where each polygon is a list of
        in-order (clockwise or counter clockwise) Points. It only one polygon,
        it must still be a list in a list, i.e. [[Point(0,0), Point(2,0),
        Point(2,1)]].
        Take advantage of processors with multiple cores by setting workers to
        the number of subprocesses you want. Defaults to 1, i.e. no subprocess
        will be started.
        Set status=False to turn off the statusbar when building.


        boundary: [[min_x, max_x], [min_y, max_y]]
        """

        self.graph = Graph(input)
        self.visgraph = Graph([])

        if boundary is not None:
            self.boundary = boundary

        points = self.graph.get_points()
        batch_size = 10 

        if workers == 1:
            for batch in [points[i:i + batch_size]
                               for i in xrange(0, len(points), batch_size)]:
                for edge in _vis_graph(self.graph, batch):
                    if boundary and not self.edge_in_boundary(edge):
                        self.visgraph.add_edge(edge)
                    elif not boundary:
                        self.visgraph.add_edge(edge)
        else:
            pool = Pool(workers)
            batches = [(self.graph, points[i:i + batch_size])
                       for i in xrange(0, len(points), batch_size)]

            results = list(pool.imap(_vis_graph_wrapper, batches))
            for result in results:
                for edge in result:
                    if boundary and not self.edge_in_boundary(edge):
                        self.visgraph.add_edge(edge)
                    elif not boundary:
                        self.visgraph.add_edge(edge)

    def find_visible(self, point):
        """Find vertices visible from point."""

        return visible_vertices(point, self.graph)

    def update(self, points, origin=None, destination=None):
        """Update visgraph by checking visibility of Points in list points."""

        for p in points:
            for v in visible_vertices(p, self.graph, origin=origin,
                                      destination=destination):
                self.visgraph.add_edge(Edge(p, v))

    # def shortest_path(self, origin, destination):
    #     """Find and return shortest path between origin and destination.

    #     Will return in-order list of Points of the shortest path found. If
    #     origin or destination are not in the visibility graph, their respective
    #     visibility edges will be found, but only kept temporarily for finding
    #     the shortest path. 
    #     """

    #     origin_exists = origin in self.visgraph
    #     dest_exists = destination in self.visgraph
    #     if origin_exists and dest_exists:
    #         return shortest_path(self.visgraph, origin, destination)
    #     orgn = None if origin_exists else origin
    #     dest = None if dest_exists else destination
    #     add_to_visg = Graph([])
    #     if not origin_exists:
    #         for v in visible_vertices(origin, self.graph, destination=dest):
    #             add_to_visg.add_edge(Edge(origin, v))
    #     if not dest_exists:
    #         for v in visible_vertices(destination, self.graph, origin=orgn):
    #             add_to_visg.add_edge(Edge(destination, v))
    #     return shortest_path(self.visgraph, origin, destination, add_to_visg)
    
    def shortest_path(self, origin, destination, blocked_edge=None, solver = "astar"):
        """Find and return shortest path between origin and destination.

        Will return in-order list of Points of the shortest path found. If
        origin or destination are not in the visibility graph, their respective
        visibility edges will be found, but only kept temporarily for finding
        the shortest path. 
        """

        origin_exists = origin in self.visgraph
        dest_exists = destination in self.visgraph
        if origin_exists and dest_exists:
            return shortest_path_glb(self.visgraph, origin, destination, add_to_visgraph = None, blocked_edge = blocked_edge, solver = solver)
        orgn = None if origin_exists else origin
        dest = None if dest_exists else destination
        add_to_visg = Graph([])
        if not origin_exists:
            for v in visible_vertices(origin, self.graph, destination=dest):
                if self.boundary and not self.edge_in_boundary(Edge(origin, v)):
                    add_to_visg.add_edge(Edge(origin, v))
                elif not self.boundary:
                    add_to_visg.add_edge(Edge(origin, v))
        if not dest_exists:
            for v in visible_vertices(destination, self.graph, origin=orgn):
                if self.boundary and not self.edge_in_boundary(Edge(destination, v)):
                    add_to_visg.add_edge(Edge(destination, v))
                elif not self.boundary:
                    add_to_visg.add_edge(Edge(destination, v))
        return shortest_path_glb(self.visgraph, origin, destination, add_to_visgraph = add_to_visg, blocked_edge = blocked_edge, solver = solver)


    def point_in_polygon(self, point):
        """Return polygon_id if point in a polygon, -1 otherwise."""

        return point_in_polygon(point, self.graph)

    def closest_point(self, point, polygon_id, length=0.001):
        """Return closest Point outside polygon from point.

        Note method assumes point is inside the polygon, no check is
        performed.
        """

        return closest_point(point, self.graph, polygon_id, length)


def _vis_graph_wrapper(args):
    try:
        return _vis_graph(*args)
    except KeyboardInterrupt:
        pass

def _vis_graph(graph, points):
    visible_edges = []
    for p1 in points:
        for p2 in visible_vertices(p1, graph, scan='half'):
            visible_edges.append(Edge(p1, p2))
    return visible_edges
