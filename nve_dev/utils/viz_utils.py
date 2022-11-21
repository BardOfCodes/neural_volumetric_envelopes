"""
1. Tools for visualizing the data - mesh, points for each cube, sdf values. 
2. Tool for extracting a mesh with marching cubes.
"""

# TODO: Test with empty envelopes
# TODO: Move cuboid definiton to different file/directory (envelope_utils)

from typing import List
import numpy as np
import random
from sdf import sdf
import os

class Cuboid :

    def __init__(self, vertices) -> None:
        assert(vertices.shape == (8, 3))

        self.vertices = vertices

        # Compute side length
        # Find vertices on shared cuboid edge
        
        x_mask = vertices[:, 0] == vertices[0, 0]
        y_mask = vertices[:, 1] == vertices[0, 1]

        edge = vertices[x_mask & y_mask]
        assert(edge.shape == (2, 3))
        self.side_length = abs(edge[0, 2] - edge[1,2]) 

        # Compute center
        self.centroid = vertices.sum(axis = 0) / 8.0

        # Compute bounds
        self.x_min = vertices[:, 0].min()
        self.x_max = vertices[:, 0].max()
        self.y_min = vertices[:, 1].min()
        self.y_max = vertices[:, 1].max()
        self.z_min = vertices[:, 2].min()
        self.z_max = vertices[:, 2].max()

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

def save_mesh(cuboids : List[np.array], output_directory = "../../results/mesh_stl", file_name = "out", num_samples_per_envelope = 2**22) :
    vertices = []
    
    for cuboid in cuboids :
        # Create neuralSDF function with params
        
        # TODO: Update sdf/sdf/d3.py to neural network
        params = {} 
        f = sdf.neuralSDF(params = params)

        # Compute list of mesh triangle vertices. (P1 P2 P3)
        points = f.generate(samples=num_samples_per_envelope)
        
        # Transform vertices from envelope_space to world_space, and store  
        points = np.array(points)
        world_space_points = cuboid.envelope_to_world(points)
        
        world_space_points = [world_space_points[i] for i in range(len(points))]

        vertices.extend(world_space_points)

    # Convert all list of triangles -> .stl file (using library's write_binary)
    os.makedirs(output_directory , exist_ok = True) 
    if not file_name.endswith('.stl') :
        file_name = file_name + ".stl"

    file_path = os.path.join(output_directory, file_name)

    sdf.write_binary_stl(file_path, vertices)
    print("Saved mesh to", file_path)


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
