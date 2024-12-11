import open3d as o3d
import copy
import numpy as np

source = o3d.io.read_point_cloud("output/combined.ply")
target = o3d.io.read_point_cloud("output/out_1_6.ply")

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    



threshold = 10
# trans_init = np.asarray([[0.862, 0.011, -0.507, 2],
#                          [-0.139, 0.967, -0.215, 2],
#                          [0.487, 0.255, 0.835, 2], [0.0, 0.0, 0.0, 1.0]])

trans_init = np.eye(4)
draw_registration_result(source, target, trans_init)


print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

combined = source.transform(reg_p2p.transformation) + target  # Apply transformation to source

# Save the combined point cloud
combined_output_path = "output/combined.ply"  # Set the output file path
o3d.io.write_point_cloud(combined_output_path, combined)
print(f"Combined point cloud saved to {combined_output_path}")