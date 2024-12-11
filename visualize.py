import open3d as o3d
import numpy as np
import copy

source = o3d.io.read_point_cloud("output/out_1_0.ply")
target = o3d.io.read_point_cloud("output/empty.ply")

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.7, 0.2, 0.2])
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