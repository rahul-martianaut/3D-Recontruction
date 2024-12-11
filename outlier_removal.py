import open3d as o3d

print("Load a ply point cloud, print it, and render it")

pcd = o3d.io.read_point_cloud("out_full.ply")

if not pcd.is_empty():
    print("Point cloud loaded successfully!")
    print(pcd)  # Prints information about the point cloud (e.g., number of points)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.8,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 0],
                                      up=[0, -1, 0])
else:
    print("Failed to load the point cloud. Please check the file path and format.")

# sample_pcd_data = o3d.data.PCDPointCloud()
# pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

# print("Downsample the point cloud with a voxel of 0.02")
# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
# o3d.visualization.draw_geometries([voxel_down_pcd],
#                                   zoom=0.8,
#                                       front=[0, 0, -1],
#                                       lookat=[0, 0, 0],
#                                       up=[0, -1, 0])


print("Every 5th points are selected")
uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
o3d.visualization.draw_geometries([uni_down_pcd],
                                  zoom=0.8,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 0],
                                      up=[0, -1, 0])

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.8,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 0],
                                      up=[0, -1, 0])
    return inlier_cloud

# Remove outliers using Radius Outlier Removal
print("Radius outlier removal")
cl, ind = uni_down_pcd.remove_radius_outlier(nb_points=50, radius=0.002)

# print("Statistical oulier removal")
# cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=50,
#                                                     std_ratio=1.0)


# Visualize and save the cleaned point cloud
inlier_cloud = display_inlier_outlier(uni_down_pcd, ind)
o3d.visualization.draw_geometries([inlier_cloud],
                                      zoom=0.8,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 0],
                                      up=[0, -1, 0])

# Save the cleaned point cloud to a file
output_file = "cleaned_point_cloud.ply"
o3d.io.write_point_cloud(output_file, inlier_cloud)
print(f"Cleaned point cloud saved as {output_file}")