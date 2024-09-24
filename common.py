from pathlib import Path
import yaml
from typing import List
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import os
import copy

        
class Point:
    def __init__(self, index: int, location: np.ndarray, normal: np.ndarray) -> None:
        self.index:int = index
        self.location:np.ndarray = location
        self.normal:np.ndarray = self.normalize(normal)

    def normalize(self, normal):
        return normal / np.linalg.norm(normal)
    
    def __eq__(self, value: 'Point') -> bool:
        if self.index == value.index:
            return True
        else:
            return False
    def __ne__(self, value: 'Point') -> bool:
        return not self.__eq__(value)

class Triangle:
    def __init__(self, pt1:Point, pt2:Point, pt3:Point,) -> None:
        """
        Index are 1x3 matrix.
        Location is a list of 3 elements. 
        """
        self.points:List[Point] = [pt1, pt2, pt3]
        self.normal:np.ndarray = None

    def set_normal(self):
        v1 = self.points[1].location - self.points[0].location
        v2 = self.points[2].location - self.points[0].location
        normal = np.cross(v1,v2)
        self.normal = normal / np.linalg.norm(normal)
    
    def left_shift(self):
        self.points = self.points[1:] + self.points[:1]
    
    def change_permutation(self):
        self.points = [self.points[0], self.points[2], self.points[1]]

class Edge:
    def __init__(self, pt_from:Point, pt_to:Point) -> None:
        """
        Index are 1x2 matrix.
        Location is a list of 2 elements. 
        """
        self.points:List[Point] = [pt_from, pt_to]
        self.triangle:Triangle = None

    def change_permutation(self):
        self.points = [self.points[1], self.points[0]]
        self.set_triangle(self.triangle)

    def set_triangle(self, triangle:Triangle): 
        """
        Set the boundary triangle that aligns with the boundary edge.
        """
        while True:
            if self.points[0] == triangle.points[0]:
                break
            else:
                triangle.left_shift() 
        if self.points[1] != triangle.points[1]:
            triangle.change_permutation()
        triangle.set_normal()
        self.triangle = triangle
    
    def __eq__(self, value: 'Edge') -> bool:
        if self.points[0].index == value.points[0].index and self.points[1].index == value.points[1].index:
            return True
        elif self.points[0].index == value.points[1].index and self.points[1].index == value.points[0].index:
            return True
        else:
            return False 


class Boundary:
    def __init__(self, list_of_edges: np.ndarray, ordered: bool) -> None:
        self.edges = np.array(list_of_edges)
        self.boundary_triangles = None
        self.vertices = None
        self.cyclic_vertices = None
        self.ordered = ordered
        self._colors = None
        self._locations = None
        self._normals = None

        if self.ordered:
            self.vertices = self.edges[:, 0]
            self.cyclic_vertices = self.__cyclic_vertices()

        self.boundary_edges: List[Edge] = []

    def __cyclic_vertices(self):
        assert self.ordered == True, "It is not a ordered edge."
        cyclic_vertices = np.zeros(len(self.vertices)+1, dtype=int)
        cyclic_vertices[:-1] = self.vertices
        cyclic_vertices[-1] = self.vertices[0]
        return cyclic_vertices

    def colors(self, string=None):
        if string == 'red':
            colors = np.array([1.0, 0.0, 0.0])
            self._colors = colors.tolist()
        elif self._colors == None:
            size = (1, 3)
            colors = np.random.uniform(size=size)
            self._colors = colors.tolist()
        return self._colors

    #def set_locations(self, locations:List[List[float]]):
    #    self._locations = np.array(locations)

    #def set_normals(self, normals:List[List[float]]):
    #    self._normals = np.array(normals)

    def set_locations(self, locations:List[List[float]]):
        #self._locations = locations[self.vertices]
        self._locations = locations

    def set_normals(self, normals:List[List[float]]):
        #self._normals = normals[self.vertices] 
        self._normals = normals

    def set_length(self):
        length = 0
        tot_points, _ = self._locations.shape
        for i in range(tot_points):
            length += np.linalg.norm(self._locations[(i+1)%tot_points]-self._locations[i,:])
        self.length = length
        
    def get_length(self):
        if self.length > -1.0:
            return self.length
        self.set_length()
        return self.length

    def set_boundary_edges(self, full_triangles: np.ndarray, full_locations: np.ndarray, full_normals: np.ndarray):
        boundary_edges = []
        for edge in self.edges:  
            triangle_index, _ = triangles_have_the_edge(edge, full_triangles)
            triangle_index = triangle_index[0]
            pt_1 = Point(triangle_index[0], full_locations[triangle_index[0]], full_normals[triangle_index[0]])
            pt_2 = Point(triangle_index[1], full_locations[triangle_index[1]], full_normals[triangle_index[1]])
            pt_3 = Point(triangle_index[2], full_locations[triangle_index[2]], full_normals[triangle_index[2]])
            triangle = Triangle(pt_1, pt_2, pt_3)

            pt_from = Point(edge[0], full_locations[edge[0]], full_normals[edge[0]])
            pt_to   = Point(edge[1], full_locations[edge[1]], full_normals[edge[1]])
            boundary_edge = Edge(pt_from, pt_to)
            boundary_edge.set_triangle(triangle)
            
            boundary_edges.append(boundary_edge)
        self.boundary_edges = boundary_edges

    def __revert_edge(self):
        new_edges = np.empty_like(self.edges)
        new_edges[:, 0] = self.edges[:, 1]
        new_edges[:, 1] = self.edges[:, 0]
        self.edges = new_edges
        self.vertices = self.edges[:, 0]
        self.cyclic_vertices = self.__cyclic_vertices()
        for edge in self.boundary_edges:
            edge.change_permutation()

    def check_orientation(self):
        sum_inner_product = 0
        sum_inner_product_inv = 0
        for edge in self.boundary_edges:
            normal0 = edge.points[0].normal
            normal1 = edge.points[1].normal

            normal_triangle = edge.triangle.normal
            inner_product     = np.dot(normal0,  normal_triangle) + np.dot(normal1,  normal_triangle)
            inner_product_inv = np.dot(normal0, -normal_triangle) + np.dot(normal1, -normal_triangle)
            sum_inner_product += inner_product
            sum_inner_product_inv += inner_product_inv
        if sum_inner_product < sum_inner_product_inv:
            self.__revert_edge()

    def __getitem__(self, key):
        return self.edges[key, :]

    def __len__(self):
        return len(self.edges[:, 0])

    @classmethod
    def from_single_list(cls, list_of_vertice: list):
        list_of_edges = list()
        for i in range(len(list_of_vertice)):
            j = (i + 1) % len(list_of_vertice)
            list_of_edges.append([list_of_vertice[i], list_of_vertice[j]])
        return cls(np.array(list_of_edges), ordered=True)

def triangles_have_the_edge(edge: np.ndarray, triangles: np.ndarray):
    l_rows, l_cols = np.where(triangles == edge[0])
    r_rows, r_cols = np.where(triangles == edge[1])
    rows = np.intersect1d(l_rows, r_rows)
    return triangles[rows, :], rows

def edges_to_lineset(mesh, edges, color):
    # From Open3D: www.open3d.org
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def create_folder(list_of_path:Path):
    for dirpath in list_of_path:
        dirpath.mkdir(parents=True, exist_ok=True)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("====================================")
    print("Configuration")
    for key, val in config.items():
        print("%10s : %s" % (key, str(val)))
    print("====================================")
    return config
    

def boundary_to_lineset(boundary:Boundary): 
    edges = boundary.edges 
    edges = np.array([[i, i+1] for i in range(len(edges))])
    edges[-1,1] = 0
    edges = o3d.utility.Vector2iVector(edges)
    ls = o3d.geometry.LineSet()
    ls.lines = edges

    vertice = boundary._locations
    ls.points = o3d.utility.Vector3dVector(vertice)

    color = boundary.colors()
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls
