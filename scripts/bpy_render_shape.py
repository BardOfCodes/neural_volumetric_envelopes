import bpy

import os

main_dir = "/home/aditya/projects/neural_volumetric_envelopes/results/single_envelopes/"
save_dir = "/home/aditya/projects/neural_volumetric_envelopes/results/single_renders/"

files = os.listdir(main_dir)
MAT_NAME = "object_mat"
newmat = bpy.data.materials.new(MAT_NAME)
newmat.use_nodes = True
node_tree = newmat.node_tree
nodes = node_tree.nodes
principled = newmat.node_tree.nodes["Principled BSDF"]
bpy.data.materials["object_mat"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.0, 1, 0.838988, 1)

# Render Settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.render.film_transparent = True
bpy.context.scene.cycles.samples = 32
bpy.context.scene.render.use_freestyle = True
bpy.context.scene.render.line_thickness = 0.5
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
# Camera Setting
camera = bpy.data.objects['Camera']
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 2.5
camera.rotation_euler[2] = 0.799219

# Light Settings
light = bpy.data.objects['Light']
light.data.type = 'SPOT'
light.data.energy = 1500

for ind, file_name in enumerate(files): 
    full_file_name = os.path.join(main_dir, file_name)
    # Load the model
    bpy.ops.import_mesh.stl(filepath=full_file_name)
    # apply material
    object_name = file_name.split(".")[0]
    object =  bpy.data.objects[object_name]
    material = bpy.data.materials[MAT_NAME]
    object.data.materials.append(material)
    
# Render
save_file_name = os.path.join(save_dir, "%d.png" % ind)
bpy.context.scene.render.filepath = save_file_name
bpy.ops.render.render(write_still = True)
    


    