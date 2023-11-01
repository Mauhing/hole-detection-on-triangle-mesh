from pathlib import Path
import yaml
from typing import List
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import os

class Boundary:
    def __init__(self, list_of_edges: np.ndarray, ordered: bool) -> None:
        self.edges = np.array(list_of_edges)
        self.vertices = None
        self.cyclic_vertices = None
        self.ordered = ordered
        self._colors = None
        self._locations = None
        self._normals = None

        if self.ordered:
            self.vertices = self.edges[:, 0]
            self.cyclic_vertices = self.__cyclic_vertices()

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

    def set_locations(self, locations:List[List[float]]):
        self._locations = np.array(locations)

    def set_normals(self, normals:List[List[float]]):
        self._normals = np.array(normals)

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

def edges_to_lineset(mesh, edges, color):
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
