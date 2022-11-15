import numpy as np
import open3d as o3d

def save_surface_points_per_grid_cell(point_sample_dir, npz_filename, grid_resolution, save_point_clouds=False):
    """
    Given a filepath to a NPZ file containing 3D sample points on the mesh's surface, saves a map 
    from grid cell IDs to sets (numpy matrices) containing on-surface points. Each grid cell that contains a zero-crossing (surface)
    is mapped to a single kx3 matrix of k 3D points on the surface in that cell. k varies depending on the area of the surface in the cell

    Note that cells are ordered in row-major order, i.e. with the formula: ID = x + y*grid_len + z*grid_len^2. The query points are in the bounding cube [-1,1]^3

    Example usage of this function and its saved npz file:
        save_surface_points_per_grid_cell("sdf_data/cuboid", "surface_points.npz", 5)
        surface_points = np.load("sdf_data/cuboid/grid_ids_to_surface_points.npz")
        print(surface_points["arr_117"]) # this retrieves the surface points at cell 118 as a Nx3 numpy matrix

    Args:
        - point_sample_dir: relative path to the directory containing this shape's 3D data. e.g. "sdf_data/cuboid"
        - npz_filename: a string filepath to a numpy NPZ file containing 3D  points. e.g. "surface_points" where "sdf_data/cuboid/surface_points.npz" contains the data
            file['points'] should return nx3 array where file is the loaded NPZ file.
        - grid_resolution: resolution of the imaginary voxel grid that subdivides the normalized space containing the shape and its SDF samples. 
            NOTE: this should be the side length of the grid, not the number of vertices per edge of the grid

    Returns: Nothing. Saves the grid cell->{k points} map to disk in the same directory as sdf filepath under the name "(...)/grid_cells_to_surface_points

    """

    # load all sample points (contains on-surface and off-surface query points for this shape)
    all_surface_sample_points = np.load(f"{point_sample_dir}/{npz_filename}")
    normalized_positions = all_surface_sample_points['surface_points'] 

    # transform positions from [0,1]^3 space to [0, grid_resolution]^3 space
    scaled_positions = grid_resolution * normalized_positions
        

    grid_cell_ID_to_surface_samples = [] # list stores grid_resolution^3 numpy arrays of variable size, one for each grid cell
    # for each grid cell, determine if it contains surface sample points and, if so, store the surface points in the map
    for grid_idx in range(grid_resolution**3):
        # find the low and high boundaries along each axis (i.e. the AABB for this grid cell)
        x_low = float(grid_idx % grid_resolution)
        x_high = x_low+1
        y_low = float((grid_idx//grid_resolution) % grid_resolution)
        y_high = y_low+1
        z_low = float((grid_idx//(grid_resolution**2)) % grid_resolution)
        z_high = z_low + 1

        # filter out surface points that are outside this grid cell
        print("="*50)
        print("x:", x_low, x_high)
        print("y:", y_low, y_high)
        print("z:", z_low, z_high)
        surface_points_in_cell = normalized_positions[
            (scaled_positions[:, 0] > x_low) &
            (scaled_positions[:, 0] < x_high) &
            (scaled_positions[:, 1] > y_low) &
            (scaled_positions[:, 1] < y_high) &
            (scaled_positions[:, 2] > z_low) &
            (scaled_positions[:, 2] < z_high)
        ] # M x 3, M can be 0 if no surface points are in this grid cell
        print(surface_points_in_cell.shape[0])

        # store the remaining surface points
        grid_cell_ID_to_surface_samples.append(surface_points_in_cell)

        if save_point_clouds and surface_points_in_cell.shape[0] > 0:
            # save point cloud PLY for visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(surface_points_in_cell)
            o3d.io.write_point_cloud(f"{point_sample_dir}/cell{grid_idx}.ply", pcd)
            print(f"saved surface point cloud at {point_sample_dir}/cell{grid_idx}.ply")

    # save the map
    np.savez(f"{point_sample_dir}/grid_ids_to_surface_points.npz", *grid_cell_ID_to_surface_samples) # unpack the arrays into arguments
    print(f"saved grid-cell-to-surface-points mapping at {point_sample_dir}/grid_ids_to_surface_points.npz")

save_surface_points_per_grid_cell("sdf_data/cuboid", "surface_points.npz", 5)
# surface_points = np.load("sdf_data/cuboid/grid_ids_to_surface_points.npz")
# print(surface_points["arr_117"].shape)