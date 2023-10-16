"""
1. Tools for visualizing the data - mesh, points for each cube, sdf values. 
2. Tool for extracting a mesh with marching cubes.
"""

from typing import List
import numpy as np
import random
from .sdf import sdf 
import os
import torch
from skimage import measure
import plyfile
import time
from tqdm import tqdm
import _pickle as cPickle
from nve_dev.dataloaders import EnvelopeDataset, NoMaskDataset
from nve_dev.dataloaders.no_mask_dl import no_mask_collate
from nve_dev.dataloaders.base_dl import worker_init_fn
import torch as th

class Cuboid :

    def __init__(self, vertices) -> None:
        assert(vertices.shape == (8, 3)) , vertices.shape

        self.vertices = vertices

        # Compute side length
        # Find vertices on shared cuboid edge
        
        x_mask = vertices[:, 0] == vertices[0, 0]
        y_mask = vertices[:, 1] == vertices[0, 1]

        edge = vertices[x_mask & y_mask]
        assert(edge.shape == (2, 3))
        self.side_length = np.array(abs(edge[0, 2] - edge[1,2]))

        # Compute center
        self.centroid = np.array(vertices.sum(axis = 0) / 8.0)
        # print("Centroid debugging", self.centroid, self.side_length)
        

        # Compute bounds
        self.x_min = vertices[:, 0].min()
        self.x_max = vertices[:, 0].max()
        self.y_min = vertices[:, 1].min()
        self.y_max = vertices[:, 1].max()
        self.z_min = vertices[:, 2].min()
        self.z_max = vertices[:, 2].max()

        # print("X/Y/Z MIN/MAX", np.array(self.x_min),np.array(self.x_max),np.array(self.y_min),np.array(self.y_max),np.array(self.z_min),np.array(self.z_max))

    def envelope_contains_point(self, points) -> bool :
        assert(len(points.shape) == 2 and points.shape[1] == 3)

        return (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) \
        & (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) \
        & (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)

    # Standardizes to [-0.5, 0.5] Cuboid
    def world_to_envelope(self, points) :
        assert(len(points.shape) == 2 and points.shape[1] == 3)
        # assert (self.envelope_contains_point(points))
        
        # Translate envelope to origin
        points = points - self.centroid
        
        # Scale based on side length
        points = points / self.side_length

        return points

    def envelope_to_world(self, points) :        
        # Scale based on side length
        points = points * self.side_length

        # Translate to centroid
        points = points + self.centroid


        return points

def save_each_envelope(model, dataloader, output_directory, num_samples_per_envelope=2 ** 15, bound_epsilon=0.1):
    # Contains some extra code for analysis.
    envelope_dict = {}
    feature_dict = {}

    for idx, batch in enumerate(dataloader):
        print(idx)
        cuboid = Cuboid(torch.squeeze(batch["envelope_vertices"]).cpu())

        # f = sdf.sphere(radius=0.5)
        if model.use_surface_normals:
            f = sdf.neuralSDF(model, batch['surface_points'], batch['surface_normals'])
        else:
            f = sdf.neuralSDF(model, batch['surface_points'])

        # Compute list of mesh triangle vertices. (P1 P2 P3)
        # I added bound epsilon, so we can get vertices on envelope boundaries (visually might improve connectiosn between envelopes?)
        min_bound = -0.5 - bound_epsilon
        max_bound = 0.5 + bound_epsilon
        points = f.generate(samples=num_samples_per_envelope, bounds = ((min_bound, min_bound, min_bound), (max_bound, max_bound, max_bound)), sparse = False)

        # Transform vertices from envelope_space to world_space, and store  
        if len(points) > 0 :
            points = np.array(points)
            # world_space_points = cuboid.envelope_to_world(points)
            world_space_points = points
            # Flip axis:
            # world_space_points -= 16
            # world_space_points = world_space_points / 10.
            world_space_points = np.stack([world_space_points[:, 0], world_space_points[:, 2], world_space_points[:, 1]], 1)
            
            world_space_points = [world_space_points[i] for i in range(len(points))]
            

            # Convert all list of triangles -> .stl file (using library's write_binary)
            os.makedirs(output_directory , exist_ok = True) 
            file_name = "_".join([str(x) for x in model.additionals['codebook_indices'][0]])
            
            if file_name in envelope_dict.keys():
                envelope_dict[file_name].append(idx)
            else:
                envelope_dict[file_name] = [idx]
            feature_dict[file_name] = model.additionals['code']
            
            if not file_name.endswith('.stl') :
                file_name = file_name + ".stl"
            file_path = os.path.join(output_directory, file_name)

            sdf.write_binary_stl(file_path, world_space_points)
            print("Saved mesh to", file_path)
            print("Number of triangles", len(world_space_points) // 3)
            
        else :
            print("No triangles found in an envelope")
            
    cPickle.dump(envelope_dict, open("results/envelope_info_32.pkl", "wb"))
    cPickle.dump(feature_dict, open("results/feature_info_32.pkl", "wb"))

    

# This renders an SDF per envelope; more useful for debugging and inspecting envelope behavior
def save_mesh_debug(model, dataloader, output_directory = "../results/mesh_stl", file_name = "out", num_samples_per_envelope = 2**22, bound_epsilon = 0.1):
    vertices = []        

    for idx, batch in enumerate(dataloader) :
        print(idx)
        cuboid = Cuboid(torch.squeeze(batch["envelope_vertices"]).cpu())

        # f = sdf.sphere(radius=0.5)
        if model.use_surface_normals:
            f = sdf.neuralSDF(model, batch['surface_points'], batch['surface_normals'])
        else:
            f = sdf.neuralSDF(model, batch['surface_points'])

        # Compute list of mesh triangle vertices. (P1 P2 P3)
        # I added bound epsilon, so we can get vertices on envelope boundaries (visually might improve connectiosn between envelopes?)
        min_bound = -0.5 - bound_epsilon
        max_bound = 0.5 + bound_epsilon
        points = f.generate(samples=num_samples_per_envelope, bounds = ((min_bound, min_bound, min_bound), (max_bound, max_bound, max_bound)), sparse = False)

        # Transform vertices from envelope_space to world_space, and store  
        if len(points) > 0 :
            points = np.array(points)
            world_space_points = cuboid.envelope_to_world(points)
            
            world_space_points = [world_space_points[i] for i in range(len(points))]

            vertices.extend(world_space_points)
        else :
            print("No triangles found in an envelope")
            # exit()

    # Convert all list of triangles -> .stl file (using library's write_binary)
    os.makedirs(output_directory , exist_ok = True) 
    if not file_name.endswith('.stl') :
        file_name = file_name + ".stl"

    file_path = os.path.join(output_directory, file_name)

    sdf.write_binary_stl(file_path, vertices)
    print("Saved mesh to", file_path)
    print("Number of triangles", len(vertices) // 3)

def save_each_plane(model, config, output_directory, num_samples_per_envelope = 2**18, special_code=1, bound_epsilon=0.1):
    # Create data loader
    # save stls to a fixed folder
    # create color indices
    
    # Instantiate DataLoader
    path = config.DATASET.PATH
    n_shapes = config.DATASET.N_SHAPES
    config.DATASET.MODE = "SINGLE"
    directories = os.listdir(path)
    directories = [x for x in directories if os.path.exists(os.path.join(path, x, "models/envelopes.pkl"))]
    directories = directories[:n_shapes]
    # Selected directories only: 
    
    
    for dir in directories:
        filepath = os.path.join(path, dir, "models/envelopes.pkl")
        
        # Instantiate DataLoader
        config.DATASET.PATH = filepath
        dataset = NoMaskDataset(config.DATASET)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=1, pin_memory=False,
                                            num_workers=0, worker_init_fn=worker_init_fn,
                                            shuffle=False)
        cur_output_directory = os.path.join(output_directory, dir)
        os.makedirs(cur_output_directory , exist_ok = True) 
        for idx, batch in enumerate(dataloader):
            print(idx)
            cuboid = Cuboid(torch.squeeze(batch["envelope_vertices"]).cpu())

            # f = sdf.sphere(radius=0.5)
            if model.use_surface_normals:
                f = sdf.neuralSDF(model, batch['surface_points'], batch['surface_normals'])
            else:
                f = sdf.neuralSDF(model, batch['surface_points'])

            # Compute list of mesh triangle vertices. (P1 P2 P3)
            # I added bound epsilon, so we can get vertices on envelope boundaries (visually might improve connectiosn between envelopes?)
            min_bound = -0.5 - bound_epsilon
            max_bound = 0.5 + bound_epsilon
            points = f.generate(samples=num_samples_per_envelope, bounds = ((min_bound, min_bound, min_bound), (max_bound, max_bound, max_bound)), sparse = False)

            # Transform vertices from envelope_space to world_space, and store  
            if len(points) > 0 :
                points = np.array(points)
                world_space_points = cuboid.envelope_to_world(points)
                world_space_points -= 16
                world_space_points = world_space_points / 10.
                world_space_points = np.stack([world_space_points[:, 0], world_space_points[:, 2], world_space_points[:, 1]], 1)
                world_space_points = [world_space_points[i] for i in range(len(points))]
                

                # Convert all list of triangles -> .stl file (using library's write_binary)
                file_name = "_".join([str(x) for x in model.additionals['codebook_indices'][0]])
                
                # Now Mark if it is the selected index
                # if file_name == special_code:
                code_ids = [x for x in model.additionals['codebook_indices'][0]]
                if special_code in code_ids:
                    file_name += "_1"
                else:
                    file_name += "_0"
                
                
                if not file_name.endswith('.stl') :
                    file_name = file_name + ".stl"
                file_path = os.path.join(cur_output_directory, file_name)
                sdf.write_binary_stl(file_path, world_space_points)
                print("Saved mesh to", file_path)
                print("Number of triangles", len(world_space_points) // 3)
                
            else :
                print("No triangles found in an envelope")
            
    
    
# This renders an SDF for the whole volume
# N - number of samples across axis; total num_points is N^3
def save_mesh(model, dataloader, output_directory = "../results/mesh_stl", file_name = "out", N = 64, max_batch = 16 ** 3) :
    start = time.time()
    os.makedirs(output_directory , exist_ok = True) 
    if not file_name.endswith('.ply') :
        file_name = file_name + ".ply"

    ply_filename = os.path.join(output_directory, file_name)

    # Change these to scale and/or offset final mesh
    offset = scale = None

    # ASSUMING (bottom, left, down) corner is (0, 0, 0)
    voxel_origin = [0, 0, 0]
    voxel_size = float(dataloader.dataset.grid_resolution) / (N - 1.0)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N   

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
 
    num_samples = N ** 3
    print("Num samples:", num_samples)

    samples.requires_grad = False

    # Pre-computing cuboid constructions
    idx_to_cuboid = {}
    for idx, batch in enumerate(dataloader) :
            cuboid = Cuboid(torch.squeeze(batch["envelope_vertices"]).cpu())
            idx_to_cuboid[idx] = cuboid

    sample_idx = 0
    while sample_idx < num_samples :
        print("sample", sample_idx)
        # world space x, y, z to sample
        sample_subset = samples[sample_idx : min(sample_idx + max_batch, num_samples), 0:3]

        # If point does not belong to any envelope, set default sdf
        samples[sample_idx : min(sample_idx + max_batch, num_samples), 3] = dataloader.dataset.grid_resolution

        for idx, batch in enumerate(dataloader) :
            cuboid = idx_to_cuboid[idx]

            # Valid samples for envelope
            envelope_mask = cuboid.envelope_contains_point(sample_subset)

            # Transform world -> envelope
            envelope_sample = cuboid.world_to_envelope(sample_subset)

            surface_points = batch['surface_points'].cuda()
            surface_normals = torch.tensor([]).cuda()

            if model.use_surface_normals:
                    surface_normals = batch['surface_normals'].cuda()
            
            p = torch.tensor(envelope_sample, device = "cuda", dtype=torch.float32)

            previous_sdf = samples[sample_idx : min(sample_idx + max_batch, num_samples), 3:]
            predicted_sdf = model.predict_sdf(p, surface_points, surface_normals).detach().cpu()

            predicted_sdf[torch.logical_not(envelope_mask)] = previous_sdf[torch.logical_not(envelope_mask)]

            samples[sample_idx : min(sample_idx + max_batch, num_samples), 3:] = predicted_sdf

        sample_idx += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename,
        offset,
        scale,
    )


# Based off https://github.com/facebookresearch/DeepSDF/blob/48c19b8d49ed5293da4edd7da8c3941444bc5cd7/deep_sdf/mesh.py
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    print(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

if __name__ == '__main__':
    print("Testing cuboid helpers")
    
    x_min = random.uniform(2.5, 4.7)
    x_max = random.uniform(5.8, 7.4)
    side_length = x_max - x_min 
    y_min = random.uniform(10.3, 11.1)
    y_max = y_min + side_length
    z_min = random.uniform(1.7, 4.3)
    z_max = z_min + side_length

    example_vertices = [
        [x_min, y_min, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
    ]

    example_vertices = np.array(example_vertices)
    c0 = Cuboid(example_vertices)

    assert(np.isclose(c0.side_length, side_length))
    assert(np.isclose(c0.x_min, x_min))
    assert(np.isclose(c0.x_max, x_max))
    assert(np.isclose(c0.y_min, y_min))
    assert(np.isclose(c0.y_max, y_max))
    assert(np.isclose(c0.z_min, z_min))
    assert(np.isclose(c0.z_max, z_max))


    x_ = np.linspace(x_min, x_max, 5)
    y_ = np.linspace(y_min, y_max, 5)
    z_ = np.linspace(z_min, z_max, 5)

    for x in x_ :
        for y in y_ :
            for z in z_ :
                sample_point = np.array([[x, y, z]])
                
                assert(c0.envelope_contains_point(sample_point))
                
                envelope_point = c0.world_to_envelope(sample_point) 
                world_point = c0.envelope_to_world(envelope_point)

                assert (np.isclose(sample_point, world_point).all())
    
    print("Testing cuboid visualization")
    v1 = [
        [-2, -2, -2],
        [-2, 0, -2],
        [-2, -2, 0],
        [-2, 0, 0],
        [0, -2, -2],
        [0, 0, -2],
        [0, -2, 0],
        [0, 0, 0],
    ]

    v1 = np.array(v1)
    c1 = Cuboid(v1)

    v2 = [
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
        [0, 2, 2],
        [2, 0, 0],
        [2, 2, 0],
        [2, 0, 2],
        [2, 2, 2],
    ]

    v2 = np.array(v2)
    c2 = Cuboid(v2)

    v3 = [
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [1, 1, 1],
    ]

    v3 = np.array(v3)
    c3 = Cuboid(v3)

    example_cuboid_list = [c1, c2, c3]
    save_mesh(example_cuboid_list, file_name = "overlapping_spheres")
