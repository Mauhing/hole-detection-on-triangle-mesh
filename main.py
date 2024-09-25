# %%
import time
import open3d as o3d
from hole_detection.hole_detection import construct_boundaries_from_mesh
from hole_detection.common import load_config

if __name__ == "__main__": 
    config = load_config('./config.yml') 
    relation = config['hole_detection']['calculate_relation']
    visualization = config['hole_detection']['show_singular_vertices']

    hole_file = './result_all_boundaries.json'
    save_relation_path = ""
    if relation:
        save_relation_path = './result_boundaries_and_holes.json'
    mesh = o3d.io.read_triangle_mesh(config['triangles_mesh_path'])
    
    # Count the number of triangles
    num_triangles = len(mesh.triangles)

    o3d.visualization.draw_geometries([mesh])
    print(type(mesh))

    print("Number of triangles in the mesh:", num_triangles)

    # Start timing
    start_time = time.time()

    # Call the function
    construct_boundaries_from_mesh(mesh, visualization = visualization, relation=relation, save_single_path=hole_file, save_relation_path=save_relation_path)

    # End timing
    end_time = time.time()

    # Calculate runtime
    runtime = end_time - start_time
    print(f"The function ran for {runtime} seconds")

# %%
