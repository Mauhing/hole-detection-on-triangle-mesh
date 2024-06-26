# Hole Detection

## Requirements
Install the necessary packages:
```
pip install numpy
pip install open3d
```

## Usage Instructions
1. Enter the path of your triangle mesh in `config.yml`.
2. Execute `main_hole_detection.py`.
3. The detected boundaries and holes will be saved as a JSON file in the current directory.
4. Utilize `main_visualizer.py` to display the detected boundaries and holes.

## Sample Data
The provided sample is a modified version of the Open3D Stanford bunny mesh.

For a detailed description of the method, visit [this link](https://www.sciencedirect.com/science/article/pii/S001044852400023X).

## Explanation of the result_boundaries_and_holes.json  
It contains [ Region1, Region2, ...]  

For each region, we have 
```
continent:[ ? x 3] (triangle indices)  
coastline:  
| - indices:  [vertices indices]  
| - locations:[? x 3] (x, y, z)  
| - normals:  [? x 3] (n_x, n_y, n_x) (Estimated normals)  
tide:  
| - 0  
|   | - indices:  [vertices indices]  
|   | - locations:[? x 3] (x, y, z)  
|   | - normals:  [? x 3] (n_x, n_y, n_x) (Estimated normals)  
|  
.  
.  
.  
| - ?  
   | - indices:  [vertices indices]  
   | - locations:[? x 3] (x, y, z)  
   | - normals:  [? x 3] (n_x, n_y, n_x) (Estimated normals)  
lake:  
| - 0  
|   | - indices:  [vertices indices]  
|   | - locations:[? x 3] (x, y, z)  
|   | - normals:  [? x 3] (n_x, n_y, n_x) (Estimated normals)  
|  
.  
.  
.  
| - ?  
    | - indices:  [vertices indices]  
    | - locations:[? x 3] (x, y, z)  
    | - normals:  [? x 3] (n_x, n_y, n_x) (Estimated normals)  
```
It shares the global coordinate as your triangle mesh.
    
## Known Issue
A bug has been identified when extracting boundary edges from `stl` files using Open3D. To circumvent this, please convert `stl` files to `ply` format using Blender, and then proceed with our implementation for loading `ply` files.
