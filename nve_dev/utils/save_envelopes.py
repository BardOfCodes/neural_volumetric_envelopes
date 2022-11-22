import numpy as np
import open3d as o3d
from collections import defaultdict
import pickle

# TODO:
# convert world space to local grid cell space for each cell (origin at x_low, y_low, z_low)
# -.5 to .5 in each cell: shift + scale points, only scale SDF
# omit non-surface envelopes

# visualize local training + surface samples as point cloud 




def save_envelope_pickle_data(point_sample_dir, surface_points_filename, training_points_filename, grid_resolution, save_point_clouds=False):
    """
    Given filepaths to NPZ files containing 3D sample points on the mesh's surface and training samples near the surface, saves a map 
    from envelope (grid cell) IDs to numpy matrices containing all the data for that envelope. 

    Note that cells are ordered in row-major order, i.e. with the formula: ID = x + y*grid_len + z*grid_len^2. The query points are in the bounding cube [-1,1]^3

    Example usage of this function and its saved npz file:
        
    save_envelope_pickle_data("sdf_data/cuboid", "surface_points.npz", "training_points.npz", 8)

    with open('sdf_data/cuboid/cuboid_envelopes.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        for envelope_id, envelope_data in loaded_dict.items():
            print(envelope_id)
            print(envelope_data["surface_points"].shape)
            print(envelope_data["training_points"].shape)
            print(envelope_data["sdf_vals_in_cell"].shape)

    Args:
        - point_sample_dir: relative path to the directory containing this shape's 3D data. e.g. "sdf_data/cuboid"
        - surface_points_filename: a string filepath to a numpy NPZ file containing 3D surface sample points. e.g. "surface_points" where "sdf_data/cuboid/surface_points.npz" contains the data
            file['surface_points'] should return nx3 array where file is the loaded NPZ file.
        - training_points_filename: same as above, but should contain sample points for training and their corresponding ground truth signed distance values. 
        - grid_resolution: resolution of the imaginary voxel grid that subdivides the normalized space containing the shape and its SDF samples. 
            NOTE: this should be the side length of the grid, not the number of vertices per edge of the grid
        - save_point_clouds: if True, a point cloud (.ply) will be saved to the same directory as the npz data for each envelope.

    Returns: Nothing. Saves the grid cell->{k points} map to disk in the same directory as sdf filepath under the name "(...)/grid_cells_to_surface_points

    """

    # load all sample points 
    training_sample_points = np.load(f"{point_sample_dir}/{training_points_filename}") # randomly permuted already
    surface_sample_points = np.load(f"{point_sample_dir}/{surface_points_filename}")
    # all points are normalized in [0,1]^3
    normalized_surface_points = surface_sample_points['surface_points'] 
    normalized_training_points = training_sample_points['points'] 
    normalized_training_gt_distances = training_sample_points['distances'] 

    # transform positions from [0,1]^3 space to [0, grid_resolution]^3 space
    scaled_surface_positions = grid_resolution * normalized_surface_points
    scaled_training_positions = grid_resolution * normalized_training_points
    scaled_gt_distances = grid_resolution * normalized_training_gt_distances
        

    envelope_ID_to_data = defaultdict(dict) # dict maps from "envelope_i" to the envelope's data (surface points, surface normals, training points, and gt distances)
    
    # for each grid cell, determine if it contains surface sample points and, if so, store the surface points in the map
    num_non_empty_envs = 0
    for envelope_idx in range(grid_resolution**3):
        # find the low and high boundaries along each axis (i.e. the AABB for this grid cell)
        x_low = float(envelope_idx % grid_resolution)
        x_high = x_low+1
        y_low = float((envelope_idx//grid_resolution) % grid_resolution)
        y_high = y_low+1
        z_low = float((envelope_idx//(grid_resolution**2)) % grid_resolution)
        z_high = z_low + 1

        # filter out points that are outside this grid cell
        surface_points_in_cell = scaled_surface_positions[
            (scaled_surface_positions[:, 0] > x_low) &
            (scaled_surface_positions[:, 0] < x_high) &
            (scaled_surface_positions[:, 1] > y_low) &
            (scaled_surface_positions[:, 1] < y_high) &
            (scaled_surface_positions[:, 2] > z_low) &
            (scaled_surface_positions[:, 2] < z_high)
        ] # M x 3, M can be 0 if no surface points are in this grid cell

        training_points_in_cell = scaled_training_positions[
            (scaled_training_positions[:, 0] > x_low) &
            (scaled_training_positions[:, 0] < x_high) &
            (scaled_training_positions[:, 1] > y_low) &
            (scaled_training_positions[:, 1] < y_high) &
            (scaled_training_positions[:, 2] > z_low) &
            (scaled_training_positions[:, 2] < z_high)
        ] # N x 3
        
        sdf_vals_in_cell = scaled_gt_distances[
            (scaled_training_positions[:, 0] > x_low) &
            (scaled_training_positions[:, 0] < x_high) &
            (scaled_training_positions[:, 1] > y_low) &
            (scaled_training_positions[:, 1] < y_high) &
            (scaled_training_positions[:, 2] > z_low) &
            (scaled_training_positions[:, 2] < z_high)
        ] # N x 3

        # ignore envelope (do not save) if no surface points exist in it
        if surface_points_in_cell.shape[0] == 0:
            continue
        print("="*50)
        print("num train points: ", training_points_in_cell.shape[0])
        print("x:", x_low, x_high)
        print("y:", y_low, y_high)
        print("z:", z_low, z_high)
        num_non_empty_envs += 1
        # reposition envelope so that all points are in [-0.5,0.5] defined relative to the center of the envelope
        
        surface_points_in_cell = (surface_points_in_cell - np.array((x_low, y_low, z_low))) - 0.5 # reposition unit envelope so that its vertex closest to the origin sits at the origin, then shift form [0,1] to [-0.5,0.5]
        training_points_in_cell = (training_points_in_cell - np.array((x_low, y_low, z_low))) - 0.5  
        # NOTE: GT distance values do NOT need to be shifted since they are translation invariant (i.e. distance to the surface does not change if we re-define the origin)
        

        # store the remaining surface points
        envelope_ID_to_data[f"envelope_{envelope_idx}"]["surface_points"] = surface_points_in_cell
        envelope_ID_to_data[f"envelope_{envelope_idx}"]["training_points"] = training_points_in_cell
        envelope_ID_to_data[f"envelope_{envelope_idx}"]["sdf_vals_in_cell"] = sdf_vals_in_cell


        if save_point_clouds and surface_points_in_cell.shape[0] > 0:
            # save point cloud PLY for visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(training_points_in_cell)
            o3d.io.write_point_cloud(f"{point_sample_dir}/cell{envelope_idx}.ply", pcd)
            print(f"saved surface point cloud at {point_sample_dir}/cell{envelope_idx}.ply")
    print(f"num nonempty: {num_non_empty_envs}")
    
    with open(f'{point_sample_dir}/cuboid_envelopes.pkl',"wb") as f:
        pickle.dump(envelope_ID_to_data, f)


save_envelope_pickle_data("sdf_data/cuboid", "surface_points.npz", "training_points.npz", 8)

with open('sdf_data/cuboid/cuboid_envelopes.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    for envelope_id, envelope_data in loaded_dict.items():
        print(envelope_id)
        print(envelope_data["surface_points"].shape)
        print(envelope_data["training_points"].shape)
        print(envelope_data["sdf_vals_in_cell"].shape)