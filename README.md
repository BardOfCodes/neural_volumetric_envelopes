# NVE - Neural Volumetric Envelopes

Given a sdf or radiance field, convert it into a set of non-overlapping volumetric envelopes. The volumetric envelopes consist of features (at their vertices), which along with a pretrained MLP approximate the field within the envelope.

## Research questions

1) Can we convert the sdf for a single mesh into a fixed lattice grid envelope?

    1) How robust are the features to noise in the lattice structure?

2) Can we learn this over a set of meshes?
3) Can we use tetrahedral envelopes instead?
4) Can we store this in a single code-book?
5) Is it better to apply this across class, or across a scene?
6) Can we generalize for different envelopes?
7) Instead of SDF should we record "surface exists instead"?

## Baseline approach

### Data processing

1. Given a mesh, fetch (x ,y, z) points with their corresponding sdf values. Also store points on the surface of the mesh.
2. Create a uniform, single resolution lattice structure around the mesh.
3. Save all cube-id with its corresponding points from step 0 : (a) Surface points, (b) sdf points for training.
   1. Optional - Store all points in canonical frame ((x,y, z) - center(corresponding_cube)). Since no scaling is involved with fixed resolution lattice, sdf points can remain as they are.

### Learning

1. We build two networks:
   1. obj-to-feature network: Takes the points in (a) as input, and predicts 8 features of size x.
   2. feature-to-sdf network: Takes the 8 features as input, and using (b) train to predict sdf(x, y, z), given points (x, y, z) as input.
2. End-to-end training!!
   1. Record validation error.

### Visualization & Logging

1. Get the mesh given the two networks, and the envelopes.
2. Logging with W&B.

### Notes

* We want end-to-end training.
* We want to avoid hardcoding the lattice structure information as plan to change that.
* Feel free to add any information!

## Advanced approach

TBD.
