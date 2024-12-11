import open3d as o3d
import numpy as np


pcd = o3d.io.read_point_cloud("output/rock/rock_clean.ply")

# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

pcd.normals = o3d.utility.Vector3dVector(np.zeros(
    (1, 3)))  # invalidate existing normals

pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)

# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# mesh = o3d.io.read_triangle_mesh(source.path)
# mesh.compute_vertex_normals()

# pcd = mesh.sample_points_poisson_disk(750)

# o3d.visualization.draw_geometries([pcd])
# alpha = 0.03
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)




# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
#     print(f"alpha={alpha:.3f}")
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#         pcd, alpha, tetra_mesh, pt_map)
#     mesh.compute_vertex_normals()
#     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)




radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))

# pcd.paint_uniform_color([1, 0, 0])  # Red color for the point cloud
# rec_mesh.paint_uniform_color([0, 0, 1])  # Blue color for the reconstructed mesh

# Draw the point cloud and the reconstructed mesh
o3d.visualization.draw_geometries([pcd, rec_mesh],
                                   )
# o3d.visualization.draw_geometries([pcd, rec_mesh])

# print('run Poisson surface reconstruction')
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd, depth=20)
# print(mesh)
# o3d.visualization.draw_geometries([mesh])

# print('remove low density vertices')
# vertices_to_remove = densities < np.quantile(densities, 0.01)
# mesh.remove_vertices_by_mask(vertices_to_remove)
# print(mesh)
# o3d.visualization.draw_geometries([mesh],
#                                   zoom=0.664,
#                                   front=[-0.4761, -0.4698, -0.7434],
#                                   lookat=[1.8900, 3.2596, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])

# output_filename = "output/reconstructed_mesh.ply"  # Change the path and filename as needed
# o3d.io.write_triangle_mesh(output_filename, mesh)

# print(f"Mesh saved to {output_filename}")