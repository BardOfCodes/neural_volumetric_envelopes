import numpy as np

def save_surface_points_per_grid_cell(sdf_filepath, grid_resolution):
    """
    Given a filepath to a NPZ file containing 3D sample points on the mesh's surface, saves a map 
    from grid cell IDs to sets (numpy matrices) containing on-surface points. Each grid cell that contains a zero-crossing (surface)
    is mapped to a single kx3 matrix of k 3D points on the surface in that cell. k varies depending on the area of the surface in the cell

    Note that cells are ordered in row-major order, i.e. with the formula: ID = x + y*grid_len + z*grid_len^2. The query points are in the bounding cube [-1,1]^3

    Args:
        - sdf_filepath: a string filepath to a numpy NPZ file containing SDF 3D query points and their corresponding distance values. 
            file['points'] should return nx3 array where file is the loaded NPZ file.
        - grid_resolution: resolution of the imaginary voxel grid that subdivides the normalized space containing the shape and its SDF samples. 
            NOTE: this should be the side length of the grid, not the number of vertices per edge of the grid

    Returns: Nothing. Saves the grid cell->{k points} map to disk in the same directory as sdf filepath under the name "(...)/grid_cells_to_surface_points

    """

    # load all sample points (contains on-surface and off-surface query points for this shape)
    all_surface_sample_points = np.load(sdf_filepath)
    normalized_positions = all_surface_sample_points['surface_points'] 

    # transform positions from [0,1]^3 space to [0, grid_resolution]^3 space
    scaled_positions = grid_resolution * normalized_positions
        

    grid_cell_ID_to_surface_samples = [] # list stores grid_resolution^3 numpy arrays of variable size, one for each grid cell
    # for each grid cell, determine if it contains surface sample points and, if so, store the surface points in the map
    for grid_idx in range(grid_resolution**3):
        # find the low and high boundaries along each axis (i.e. the AABB for this grid cell)
        x_low = float(grid_idx % grid_resolution)
        x_high = x_low+1
        y_low = (grid_idx//grid_resolution) % grid_resolution
        y_high = y_low+1
        z_low = (grid_idx//(grid_resolution**2)) % grid_resolution
        z_high = z_low + 1

        # filter out surface points that are outside this grid cell
        print(x_low, x_high)
        print(scaled_positions.shape)
        surface_points_in_cell = normalized_positions[
            (scaled_positions[:, 0] > 2.5) &
            (scaled_positions[:, 0] < x_high) &
            (scaled_positions[:, 1] > y_low) &
            (scaled_positions[:, 1] < y_high) &
            (scaled_positions[:, 2] > z_low) &
            (scaled_positions[:, 2] < z_high)
        ] # M x 3, M can be 0 if no surface points are in this grid cell
        print(surface_points_in_cell.shape)
        exit()
        # store the remaining surface points
        grid_cell_ID_to_surface_samples.append(surface_points_in_cell)

    exit()
    # save the map
    np.savez(prefix + "dense_all_points_manifold.npz", *grid_cell_ID_to_surface_samples) # unpack the arrays into arguments

save_surface_points_per_grid_cell("sdf_data/cuboid/surface_points.npz", 5)