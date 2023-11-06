from array import array
from copy import deepcopy
from tkinter import W
import numpy as np
import json
import yaml
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from typing import List
import open3d as o3d
from common import load_config, Boundary, boundary_to_lineset
from main_hole_detection import get_length_of_boundary

def load_json_data(json_file_path):
    with open(json_file_path, 'r') as j:
        list_of_dictionaries = json.loads(j.read())
    return list_of_dictionaries


def get_boundary_from_dict(dict_):
    boundary_i = dict_['indices']
    boundary_v = dict_['locations']
    boundary_n = dict_['normals']

    boundary = Boundary.from_single_list(boundary_i)
    boundary.set_locations(boundary_v)
    boundary.set_normals(boundary_n)
    return boundary


def edges_to_lineset(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def visualize_boundaries(boundaries: List[Boundary]):
    ''' Visualize_boundary_and_mesh
    '''
    geoms = []
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame())

    for boundary in boundaries:
        geoms.append(boundary_to_lineset(boundary))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)


def unpack_region(region):
    main_boundary = get_boundary_from_dict(region['coastline'])
    main_triangles = np.array(region['continent'], dtype=int) 
    tide_list = [get_boundary_from_dict(tide) for tide in region['tide']]
    lake_list = [get_boundary_from_dict(lake) for lake in region['lake']]
    return main_boundary, tide_list, lake_list, main_triangles
    

def get_all_boundaries_and_holes(list_of_regions, vertices):
    all_boundaries = list()
    
    print(f'number_of_main boundaries {len(list_of_regions)}')
    for index, region in enumerate( list_of_regions):
        print('=='*8)
        main_boundary, tide_list, lake_list, _ = unpack_region(region)
        all_boundaries += [main_boundary] + tide_list + lake_list
        print(f'Main boundary {index+1}')
        print(f'Main boundary length {get_length_of_boundary(main_boundary, vertices)}')
        print(f'Number of tide {len(tide_list)}')
        print(f'Number of lake {len(lake_list)}')

    return all_boundaries

def visualize_all_boundaries(list_of_regions):
    visualize_boundaries(get_all_boundaries_and_holes(list_of_regions))


def visualize_all_main_boundaries(list_of_regions):
    all_boundaries = list()
    for region in list_of_regions:
        main_boundary = get_boundary_from_dict(region['main_boundary'])
        all_boundaries += [main_boundary]
    visualize_boundaries(all_boundaries)


def visualize_nm_vertices(boundaries, mesh):
    ''' Visualize_boundary_and_mesh
    '''

    mesh.compute_vertex_normals()
    geoms = [mesh]
    #geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame())

    for boundary in boundaries:
        boundary_ = o3d.utility.Vector2iVector(boundary.edges)
        geoms.append(edges_to_lineset(
            mesh, boundary_, (1, 0, 0)))

    verts = np.asarray(mesh.get_non_manifold_vertices())
    pcl = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
    pcl.paint_uniform_color((0, 0, 1))
    geoms.append(pcl)
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

def visualize_simple_boundaries_and_mesh(boundaries, mesh):
    ''' Visualize_boundary_and_mesh
    '''

    mesh.compute_vertex_normals()
    geoms = [mesh]
    #geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame())

    for boundary in boundaries:
        boundary_ = o3d.utility.Vector2iVector(boundary.edges)
        geoms.append(edges_to_lineset(
            mesh, boundary_, boundary.colors()))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

    geoms = []
    for boundary in boundaries:
        boundary_ = o3d.utility.Vector2iVector(boundary.edges)
        #geoms.append(hole_pose(boundary, locations, normals))
        geoms.append(edges_to_lineset(
            mesh, boundary_, boundary.colors()))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)

def visualize_boundaries_and_mesh(list_of_regions, mesh):
    ''' Visualize_boundary_and_mesh
    '''
    boundaries = get_all_boundaries_and_holes(list_of_regions, np.array(mesh.vertices))

    mesh.compute_vertex_normals()
    geoms = [mesh]
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame())

    for boundary in boundaries:
        boundary_ = o3d.utility.Vector2iVector(boundary.edges)
        geoms.append(edges_to_lineset(
            mesh, boundary_, boundary.colors()))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)

    geoms = []
    for boundary in boundaries:
        boundary_ = o3d.utility.Vector2iVector(boundary.edges)
        #geoms.append(hole_pose(boundary, locations, normals))
        geoms.append(edges_to_lineset(
            mesh, boundary_, boundary.colors()))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)


def visualize_one_by_one(list_of_regions, mesh): 
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)
    for index, region in enumerate( list_of_regions):
        main_boundary, tide_list, lake_list, main_triangles = unpack_region(region)
        boundaries = [main_boundary] + tide_list + lake_list
        local_mesh = copy.deepcopy(mesh)
        local_mesh.triangles = o3d.utility.Vector3iVector(main_triangles)
        local_mesh.compute_vertex_normals()
        
        geoms = []
        for boundary in boundaries:
            boundary_ = o3d.utility.Vector2iVector(boundary.edges)
            #geoms.append(hole_pose(boundary))
            geoms.append(edges_to_lineset(
                mesh, boundary_, boundary.colors()))
        print(f'Coastline number {index +1}')
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)

        geoms = [local_mesh]
        for boundary in boundaries:
            boundary_ = o3d.utility.Vector2iVector(boundary.edges)
            geoms.append(edges_to_lineset(
                mesh, boundary_, boundary.colors()))
        print(f'The continent of coastline number {index +1}')
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)

        geoms = []
        for boundary in tide_list + [main_boundary]:
            boundary_ = o3d.utility.Vector2iVector(boundary.edges)
            geoms.append(edges_to_lineset(
                mesh, boundary_, boundary.colors()))
        print(f"The tide-hole(s) with of coastline number {index +1}")
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)

        geoms = []
        for boundary in lake_list + [main_boundary]:
            boundary_ = o3d.utility.Vector2iVector(boundary.edges)
            geoms.append(edges_to_lineset(
                mesh, boundary_, boundary.colors()))
        print(f"The lake-hole(s) with of coastline number {index + 1}")
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=False)
        if index == 2:
            print('Break, we only show the first three')
            break
    

if __name__ == "__main__":

    config = load_config('./config.yml')
    mesh = o3d.io.read_triangle_mesh(config['triangles_mesh_path'])
    
    show_all_boundaries = config['visualizer']['show_all_boundaries']
    show_relations = config['visualizer']['show_relations']

    if show_all_boundaries == False and show_relations == False:
        print('Nothing to show. Please change the config file.')

    if show_all_boundaries:
        simply_json_file_path = Path('./result_all_boundaries.json')
        list_of_boundaries = load_json_data(simply_json_file_path)
        bb = []
        for dict_ in list_of_boundaries:
            bb.append(get_boundary_from_dict(dict_)) 
        visualize_nm_vertices(bb, mesh)
        visualize_simple_boundaries_and_mesh(bb, mesh)

    if show_relations:
        json_file_path = Path('./result_boundaries_and_holes.json')
        list_of_regions = load_json_data(json_file_path)
        visualize_one_by_one(list_of_regions, mesh) 