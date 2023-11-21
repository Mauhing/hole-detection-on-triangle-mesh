# %%
import time
import copy
import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List
from common import load_config, Boundary, create_folder


def get_pcd_path(file_path):
    pre, ext = os.path.splitext(file_path)
    ply_file_path = pre + '_pcd.ply'
    return ply_file_path


def get_pcd_norm_estim_path(file_path):
    pre, ext = os.path.splitext(file_path)
    ply_file_path = pre + '_pcd_norm_estim.ply'
    return ply_file_path


def change_point_color(pcd, point_num, t):
    all_points_colors = np.asarray(pcd.colors)
    all_points_colors[point_num] = [1.0, 0.0, t]
    pcd.colors = o3d.utility.Vector3dVector(all_points_colors)
    return pcd


def edges_to_lineset(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def non_manifold_element(mesh, visualization=False, simplify=None):
    if simplify is not None:
        mesh = mesh.simplify_quadric_decimation(simplify)
    mesh.compute_vertex_normals()
    geoms = [mesh]
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    geoms.append(edges_to_lineset(mesh, edges, (1, 0, 0)))

    verts = np.asarray(mesh.get_non_manifold_vertices())
    pcl = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
    pcl.paint_uniform_color((0, 0, 1))
    geoms.append(pcl)
    if visualization:
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)
    return np.array(edges), np.array(verts)


def check_properties(mesh):
    # From: http://www.open3d.org/docs/release/tutorial/geometry/mesh.html
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    #edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    #vertex_manifold = mesh.is_vertex_manifold()
    #self_intersecting = mesh.is_self_intersecting()
    #watertight = mesh.is_watertight()
    #orientable = mesh.is_orientable()
    assert edge_manifold, 'The triangle mesh is not edge-manifold.'


def connectivity_and_vertice_manifoldness(edges_array, v_next):
    rows, cols = np.where(edges_array == v_next)
    if rows.shape[0] > 2:
        is_vertice_non_manifold = True
    elif rows.shape[0] == 2:
        is_vertice_non_manifold = False
    else:
        raise Exception(
            "The vertice can only be found once in edeges set. This is not aligh with our assumption.")
    return rows, cols, is_vertice_non_manifold


def get_index_opposite_vertice(index_from, index_to):
    temp = [0, 1, 2]
    temp.remove(index_from)
    temp.remove(index_to)
    return temp[0]


def find_triangles_given_a_boundary_edge(triangles, v_from, v_to):
    rows_tri_to, cols_tri_to = np.where(triangles == v_to)
    rows_tri_from, cols_tri_from = np.where(triangles == v_from)
    intersected_indices, indices_from, indices_to = np.intersect1d(
        rows_tri_from, rows_tri_to, return_indices=True)
    assert len(intersected_indices) == 1, "No triangle or more than one triangle connect to this edge. Are you sure it is a boundary edge."
    row_tri_index = intersected_indices[0]
    index_from_vertice = cols_tri_from[indices_from[0]]
    index_to_vertice = cols_tri_to[indices_to[0]]
    index_opposite_vertice = get_index_opposite_vertice(
        index_from_vertice, index_to_vertice)
    next_triangle = triangles[row_tri_index, :]
    opposite_vertice_number = next_triangle[index_opposite_vertice]
    return next_triangle, opposite_vertice_number


def find_triangles_given_a_common_edge(triangles, last_triangle, v_to, v_temp):
    rows_tri_to, cols_tri_to = np.where(triangles == v_to)
    rows_tri_from, cols_tri_from = np.where(triangles == v_temp)
    intersected_indices, indices_from, indices_to = np.intersect1d(
        rows_tri_from, rows_tri_to, return_indices=True)
    if len(intersected_indices) != 2:
        print(f'Interseted_indeces: {len(intersected_indices)}')
        print(f'v_to: {v_to}')
        print(f'v_temp: {v_temp}')
    assert len(
        intersected_indices) == 2, f"More or less than 2 triangles connect to this edge. {len(intersected_indices)}"
    if np.array_equal(triangles[intersected_indices[0]], last_triangle):
        row_tri_index = intersected_indices[1]
        index_from_vertice = cols_tri_from[indices_from[1]]
        index_to_vertice = cols_tri_to[indices_to[1]]
    else:
        row_tri_index = intersected_indices[0]
        index_from_vertice = cols_tri_from[indices_from[0]]
        index_to_vertice = cols_tri_to[indices_to[0]]
    index_opposite_vertice = get_index_opposite_vertice(
        index_from_vertice, index_to_vertice)
    next_triangle = triangles[row_tri_index, :]
    opposite_vertice_number = next_triangle[index_opposite_vertice]
    return next_triangle, opposite_vertice_number


def is_number_in_array(candidate_edges, number):
    rows, cols = np.where(candidate_edges == number)
    if len(rows) == 1 and len(cols) == 1:
        return True, rows
    elif len(rows) == 0:
        return False, None
    else:
        raise ValueError(
            'More than one candidate edge are found. It can not be.')


def find_the_boundary_edge_on_non_manifold_vertice(mesh, edges_array, v_from, v_to, rows):
    candidate_edges = edges_array[rows, :]
    triangles = np.array(mesh.triangles)
    # Initial phase: boundary triangle
    triangle, opposite_vertice_number = find_triangles_given_a_boundary_edge(
        triangles, v_from, v_to)
    found, new_rows = is_number_in_array(
        candidate_edges, opposite_vertice_number)
    # Looping phase
    while found == False:
        triangle, opposite_vertice_number = find_triangles_given_a_common_edge(
            triangles, triangle, v_to, opposite_vertice_number)
        found, new_rows = is_number_in_array(
            candidate_edges, opposite_vertice_number)
    next_connected_boundary_edge = candidate_edges[new_rows[0], :]
    row = rows[new_rows[0]]
    return next_connected_boundary_edge, opposite_vertice_number, row


def extract_a_boundary_from_edges_set(edges_array, mesh):
    # Initial edge_row
    row = 0
    v_start, v_next = edges_array[row]
    v_from = v_start
    edge = edges_array[row]
    start_edge = edges_array[row]

    boundary_ordered = [edge]
    while True:
        rows, cols, is_vertice_non_manifold = connectivity_and_vertice_manifoldness(
            edges_array, v_next)
        if is_vertice_non_manifold == True:
            edge, v_o, row = find_the_boundary_edge_on_non_manifold_vertice(
                mesh, edges_array, v_from, v_next, rows)
            v_from = v_next
            v_next = v_o
        else:
            edge1 = edges_array[rows[0], :]
            edge2 = edges_array[rows[1], :]
            edge = next_edge(edge1, edge2, [v_from, v_next])

            v_from = v_next
            v_next = next_vertice(v_next, edge)

        if v_from == start_edge[0] and v_next == start_edge[1]:
            break
        else:
            boundary_ordered.append([v_from, v_next])
    return boundary_ordered


def next_vertice(vertice: int, edge: List[int]):
    if vertice == edge[0]:
        return edge[1]
    elif vertice == edge[1]:
        return edge[0]
    else:
        raise Exception("Vertice can not be found in the edge")


def next_edge(edge1: List[int], edge2: List[int], edge_previous: List[int]):
    same_1 = is_same_edge(edge1, edge_previous)
    same_2 = is_same_edge(edge2, edge_previous)
    if same_1 is False and same_2 is True:
        return edge1
    elif same_1 is True and same_2 is False:
        return edge2
    else:
        raise Exception(
            "There should be aleast one edge equal to edge_previous")


def is_same_edge(edge1: List[int], edge2: List[int]):
    if edge1[0] == edge2[0] and edge1[1] == edge2[1]:
        return True
    if edge1[0] == edge2[1] and edge1[1] == edge2[0]:
        return True
    return False


def remove_boundary_from_edge_set(bonndary: List[List[int]], edges_array: np.ndarray):
    for edge in bonndary:
        edges_array = remove_edge_from_edge_set(edge, edges_array)
    return edges_array


def remove_edge_from_edge_set(edge: List[int], edges_array: np.ndarray):
    rows_from, _ = np.where(edges_array == edge[0])
    rows_to, _ = np.where(edges_array == edge[1])

    common_indice = np.intersect1d(rows_from, rows_to)
    # Sanity check
    assert len(common_indice) == 1, f"A edge should only be found once in the edge set. But it is not the case now, something is wrong."

    edges_array = np.delete(edges_array, common_indice[0], axis=0)
    return edges_array


def seperate_non_manifold_edge(edges_array, mesh):
    boundaries = list()
    while edges_array.size != 0:
        #print(f'edge_array {edges_array.size}')
        boundary = extract_a_boundary_from_edges_set(
            edges_array, mesh)
        edges_array = remove_boundary_from_edge_set(boundary, edges_array)
        boundaries.append(Boundary(boundary, ordered=True))
    return boundaries


def construct_boundaries_from_nm_edges(edges, mesh, visualization=False):
    '''Returns mainly a ordered list of boundaries. Each boundary has edges that
    can be connected with a single mesh. Remark that the ordered of the boundary
    do not imply anything about the orientation.
    '''
    mesh.compute_vertex_normals()
    geoms = [mesh]
    boundaries_list = seperate_non_manifold_edge(
        edges, mesh)
    for index, boundary in enumerate(boundaries_list):
        colors = boundary.colors()
        boundary = o3d.utility.Vector2iVector(np.asarray(boundary))
        geoms.append(edges_to_lineset(mesh, boundary, colors))
    if visualization:
        print("Show different boundary with different color.")
        #o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)
    return boundaries_list


def get_length_of_boundary(boundary: Boundary, vertices: np.ndarray):
    length = 0
    for edge_ordered in boundary:
        v_from, v_to = edge_ordered
        length_vector = vertices[v_to, :] - vertices[v_from, :]
        length += np.linalg.norm(length_vector)
    return length


def get_length_of_boundaries(boundaries_ordered: List[Boundary], vertices: np.ndarray):
    boundaries_length = list()
    for boundary_ordered in boundaries_ordered:
        # print(type(boundary_ordered))
        #rows_lenght, cols_length = boundary_ordered
        length = get_length_of_boundary(boundary_ordered, vertices)
        boundaries_length.append(length)
    return boundaries_length


def choose_model_boundary_with_max_length(boundaries_ordered: List[Boundary], vertices: np.ndarray):
    boundaries_lenght = get_length_of_boundaries(boundaries_ordered, vertices)
    argmax = np.argmax(boundaries_lenght)
    main_boundary = copy.deepcopy(boundaries_ordered[argmax])
    other_boundaries = copy.deepcopy(boundaries_ordered)
    del other_boundaries[argmax]
    return main_boundary, other_boundaries


def create_different_colors(boundaries: List[np.ndarray]):
    return np.random.uniform(size=(len(boundaries), 3)).tolist()


def triangles_have_the_edge(edge: np.ndarray, triangles: np.ndarray):
    l_rows, l_cols = np.where(triangles == edge[0])
    r_rows, r_cols = np.where(triangles == edge[1])
    rows = np.intersect1d(l_rows, r_rows)
    return triangles[rows, :], rows


def find_lake_holes(main_triangles: np.ndarray, remaining_boundaries_ordered: List[Boundary]):
    lake_holes = list()
    new_remaining_boundaries_ordered = list()
    for boundary in remaining_boundaries_ordered:
        _, rows = triangles_have_the_edge(boundary[0], main_triangles)
        if len(rows) > 0:
            lake_holes.append(boundary)
        else:
            new_remaining_boundaries_ordered.append(boundary)
    return lake_holes, new_remaining_boundaries_ordered


def find_holes(main_triangles: np.ndarray, remaining_boundaries_ordered: List[Boundary]):
    holes = list()
    new_remaining_boundaries_ordered = list()
    for boundary in remaining_boundaries_ordered:
        _, rows = triangles_have_the_edge(boundary[0], main_triangles)
        if len(rows) > 0:
            holes.append(boundary)
        else:
            new_remaining_boundaries_ordered.append(boundary)
    return holes, new_remaining_boundaries_ordered


def classify_holes(main_boundary: Boundary, holes: List[Boundary]):
    lake_holes = list()
    tide_holes = list()
    for hole in holes:
        intersection= np.intersect1d(main_boundary.vertices, hole.vertices)
        has_intersection = len(intersection) > 0
        if has_intersection:
            tide_holes.append(hole)
        else:
            lake_holes.append(hole)
    return tide_holes, lake_holes

    
def find_main_triangles(triangles: np.ndarray, main_boundary_ordered: Boundary):
    """
    This methods use triangles
    """
    remain_triangles = np.copy(triangles)
    model_triangles = list()
    inquiry_triangles, remain_triangles = pop_triangle_has_the_edge(
        main_boundary_ordered[0], remain_triangles)

    while True:
        inquiry_triangle = inquiry_triangles.pop()
        neighbour_triangles, remain_triangles = pop_neighbour_triangles(
            inquiry_triangle, remain_triangles)
        inquiry_triangles = inquiry_triangles + neighbour_triangles
        model_triangles.append(inquiry_triangle)
        if len(inquiry_triangles) == 0:
            break
    return np.array(model_triangles)


def pop_triangle_has_the_edge(edge: np.ndarray, remain_triangles: np.ndarray):
    seed_triangles, rows = triangles_have_the_edge(edge, remain_triangles)
    inquiry_triangles = list()
    for row in seed_triangles:
        inquiry_triangles.append(row)
    remain_triangles = np.delete(remain_triangles, rows, axis=0)
    return inquiry_triangles, remain_triangles


def pop_neighbour_triangles(inquiry_triangle: np.ndarray, remain_triangles: np.ndarray):
    neighbour_triangles = list()
    for i in range(3):
        edge = [inquiry_triangle[i], inquiry_triangle[(i+1) % 3]]
        neighbour_triangle, remain_triangles = pop_triangle_has_the_edge(
            edge, remain_triangles)
        neighbour_triangles = neighbour_triangles + neighbour_triangle
        assert len(neighbour_triangle) < 2, "There should only max one triangle"
    return neighbour_triangles, remain_triangles


def save_boundaries_as_json(holes: List[Boundary], vertices, normals, save_single_path):
    holes_list = list()
    for index, hole in enumerate(holes):
        hole_vertices = hole.vertices
        hole_locations = vertices[hole_vertices]
        hole_normals = normals[hole_vertices]
        dict_ = {"hole_number": index,
                 "indices": hole_vertices.tolist(), 
                 "locations": hole_locations.tolist(), 
                 "normals": hole_normals.tolist()
                 }
        holes_list.append(dict_)

    hole_file = save_single_path
    with open(hole_file, 'w') as f:
        json.dump(holes_list, f)
    print(f'Simple boundaries saved: {hole_file}')


def create_dict_boundary_and_holes(boundaries: List[Boundary],
                                   tide_pool_holes: List[List[Boundary]],
                                   lake_holes: List[List[Boundary]],
                                   main_triangles_list: np.ndarray,
                                   vertices: np.ndarray,
                                   normals: np.ndarray):
    dict_list = list()
    for index_b, boundary in enumerate(boundaries):
        dict_region = dict()
        boundary_vertices = boundary.vertices
        boundary_locations = vertices[boundary_vertices]
        boundary_normals = normals[boundary_vertices]
        dict_boundary = {"indices": boundary_vertices.tolist(
        ), "locations": boundary_locations.tolist(), "normals": boundary_normals.tolist()}
        dict_region['coastline'] = dict_boundary

        dict_region['continent'] = main_triangles_list[index_b].tolist()

        tide_pools_list = list()
        for index, tide_pool_hole in enumerate(tide_pool_holes[index_b]):
            tide_pool_vertices = tide_pool_hole.vertices
            tide_pool_locations = vertices[tide_pool_vertices]
            tide_pool_normals = normals[tide_pool_vertices]
            dict_tide_pool_ = {"indices": tide_pool_vertices.tolist(), "locations": tide_pool_locations.tolist(), "normals": tide_pool_normals.tolist()}
            tide_pools_list.append(dict_tide_pool_)
        dict_region['tide'] = tide_pools_list

        lakes_list = list()
        for index, lake_hole in enumerate(lake_holes[index_b]):
            lake_vertices = lake_hole.vertices
            lake_locations = vertices[lake_vertices]
            lake_normals = normals[lake_vertices]
            dict_lake_ = {"indices": lake_vertices.tolist(
            ), "locations": lake_locations.tolist(), "normals": lake_normals.tolist()}
            lakes_list.append(dict_lake_)
        dict_region['lake'] = lakes_list

        dict_list.append(dict_region)
    return dict_list


def has_repeated_vertice(boundary: Boundary):
    for index_l in range(len(boundary)):
        vertice_left = boundary[index_l][0]
        # potensial to optimize the same index will not be next to each another.
        for index_r in range(index_l + 1, len(boundary)):
            vertice_right = boundary[index_r][0]
            if vertice_left == vertice_right:
                return True, index_l, index_r
    return False, 0, 0


def decompose_circuit_to_circles(boundary: Boundary):
    statement, index1, index2 = has_repeated_vertice(boundary)
    if statement == False:
        return [boundary]
    else:
        boundary1 = Boundary.from_single_list(
            boundary.vertices[:index1].tolist() + boundary.vertices[index2:].tolist())
        boundary2 = Boundary.from_single_list(boundary.vertices[index1:index2])
    return decompose_circuit_to_circles(boundary1) + decompose_circuit_to_circles(boundary2)


def decompose_circuit_to_circles_all(boundaries: List[Boundary]):
    simple_boundaries = list()
    for boundary in boundaries:
        simple_boundaries = simple_boundaries + decompose_circuit_to_circles(boundary)
    return simple_boundaries


def __save_point_cloud_previous(list_of_messages, project_name: str):
    path = Path('./outputs/' + project_name)
    create_folder([path])

    json_path = path / (project_name + '_holes'+'.json')
    with open(json_path, 'w') as f:
        json.dump(list_of_messages, f)
        print("saved in JSON file as " + str(json_path))


def __save_point_cloud(list_of_messages, save_relation_path:str):
    path = save_relation_path
    with open(path, 'w') as f:
        json.dump(list_of_messages, f)
        print("Relation saved in JSON file as " +path)


def number_single_boundaries_has_singular(boundaries: List[Boundary], singular_vertice: np.ndarray):
    count = 0
    for boundary in boundaries:
        intersect = np.intersect1d(boundary.vertices, singular_vertice) 
        if len(intersect) > 0:
            count += 1
    
    print(f'Total number of simple boundaries: {len(boundaries)}')
    print(f'Total number of simple boundaries that has singular vertex: {count}')
    print(f'Ratio {count/len(boundaries)}')


def construct_boundaries_from_mesh(mesh, visualization = False, relation=False, save_single_path="", save_relation_path=""):

    #mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    check_properties(mesh)
    #print('Finish check manifoldness')

    nm_edges, nm_vertices = non_manifold_element(
        mesh, visualization=visualization, simplify=None)
    #print(f"Does the mesh have triangles normal: {mesh.has_triangle_normals()}")
    # nm_vertices did not get used.

    vertices = np.array(mesh.vertices)
    normals = np.array(mesh.vertex_normals)
    triangles = np.array(mesh.triangles)

    # Construct boundary from non manifold edge
    #print('Constructing boundaries')
    all_boundaries_ordered = construct_boundaries_from_nm_edges(
        nm_edges, mesh, visualization=visualization)

    remaining_boundaries_ordered = all_boundaries_ordered  # init while loop
    #print('Decomposing boundaries')
    remaining_boundaries_ordered = decompose_circuit_to_circles_all(remaining_boundaries_ordered)
    all_single_boundaries = copy.copy(remaining_boundaries_ordered)

    #number_single_boundaries_has_singular(remaining_boundaries_ordered, nm_vertices)

    main_boundaries_list = list()
    tide_pool_holes_list = list()
    lake_holes_list = list()
    main_triangles_list = list()
    
    #print(f'Total boundaries {len(remaining_boundaries_ordered)}')
    dict_ = None
    if relation:
        print('Calculating relations')
        while len(remaining_boundaries_ordered) > 0:
            print(f'Remaining boundaries {len(remaining_boundaries_ordered)} to be calculated.')

            main_boundary_ordered, remaining_boundaries_ordered = choose_model_boundary_with_max_length(
                remaining_boundaries_ordered, vertices)

            main_triangles = find_main_triangles(triangles, main_boundary_ordered)

            holes_, remaining_boundaries_ordered = find_holes(
                main_triangles, remaining_boundaries_ordered)

            tide_pool_holes, lake_holes = classify_holes(main_boundary_ordered, holes_)

            locations = vertices

            mesh_temp = copy.deepcopy(mesh)
            mesh_temp.triangles = o3d.utility.Vector3iVector(main_triangles)

            main_boundaries_list.append(main_boundary_ordered)
            main_triangles_list.append(main_triangles)
            tide_pool_holes_list.append(tide_pool_holes)
            lake_holes_list.append(lake_holes)


        dict_ = create_dict_boundary_and_holes(main_boundaries_list, tide_pool_holes_list, lake_holes_list, main_triangles_list,
                                            locations, normals)


    if save_single_path != "": 
        save_boundaries_as_json(all_single_boundaries, vertices, normals, save_single_path)
    if save_relation_path !="":
        __save_point_cloud(dict_, save_relation_path) 
    return all_boundaries_ordered, dict_


# %%
if __name__ == "__main__": 
    config = load_config('./config.yml') 
    relation = config['hole_detection']['calculate_relation']
    visualization = config['hole_detection']['show_singular_vertices']

    hole_file = './result_all_boundaries.json'
    save_relation_path = ""
    if relation:
        save_relation_path = './result_boundaries_and_holes.json'
    mesh = o3d.io.read_triangle_mesh(config['triangles_mesh_path'])
    construct_boundaries_from_mesh(mesh, visualization = False, relation=relation, save_single_path=hole_file, save_relation_path=save_relation_path)
