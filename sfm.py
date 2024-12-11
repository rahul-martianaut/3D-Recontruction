import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import cv2 
import numpy as np 
import collections
import glob
import os
from utils import GetImageMatches, GetAlignedMatches, GetTriangulatedPts, ComputeEpiline, drawlines, pts2ply

# #auto-reloading external modules
# %load_ext autoreload
# %autoreload 2

img1 = cv2.imread('fountain/0001.png')
img2 = cv2.imread('fountain/0002.png')

#Converting from BGR to RGB format
img1 = img1[:,:,::-1]
img2 = img2[:,:,::-1]

#NOTE: you can adjust appropriate figure size according to the size of your screen
f, (ax0, ax1) = plt.subplots(1,2,figsize=(9,4))
ax0.imshow(img1)
ax1.imshow(img2)
plt.show()


#Getting SIFT/SURF features for image matching (this might take a while)
kp1,desc1,kp2,desc2,matches=GetImageMatches(img1,img2)
#kp1,desc1,kp2,desc2,matches, img1pts, img2pts = feature_matching(img1, img2)

#Aligning two keypoint vectors
img1pts,img2pts,img1idx,img2idx=GetAlignedMatches(kp1,desc1,kp2,desc2,matches)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    flags = 2)

plt.imshow(cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params))

F,mask=cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_7POINT)#, ransacReprojThreshold = 3, confidence=0.8) #FM_LMEDS, FM_8POINT, FM_RANSAC
mask=mask.astype(bool).flatten()
print(F)

#Inliers // Optional
img1pts = img1pts[mask==True]
img2pts = img2pts[mask==True]
mask = len(img1pts) * [True] ### We need the match matrix to be the same size of the number of points


lines2=ComputeEpiline(img1pts[mask],1,F)
lines1=ComputeEpiline(img2pts[mask],2,F)



epilines1, epilines2 = drawlines(img2,img1,lines2,img2pts[mask],img1pts[mask],drawOnly=10,linesize=18,circlesize=10)
epilines3, epilines4 = drawlines(img1,img2,lines1,img1pts[mask],img2pts[mask],drawOnly=10,linesize=18,circlesize=10)

epilines12 = np.concatenate((epilines2, epilines1), axis=1)
plt.imshow(epilines12)
plt.show()

epilines34 = np.concatenate((epilines3, epilines4), axis=1)
plt.imshow(epilines34)
plt.show()

epilines = np.concatenate((epilines3, epilines1), axis=1)

plt.imshow(epilines)
plt.show()

K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])


topologies = collections.OrderedDict()
topologies['360'] = tuple(zip((0,1,2,3,4,5,6,7,8,9,10,11),
                          (1,2,3,4,5,6,7,8,9,10,11,0)))

topologies['overlapping'] = tuple(zip((0,1,2,3,4,5,6,7,8,9),
                          (1,2,3,4,5,6,7,8,9,10)))

topologies['adjacent'] = tuple(zip((0,2,4,6,8,10),
                     (1,3,5,7,9,11)))

topologies['skipping_1'] = tuple(zip((0,3,6,9),
                 (1,4,7,10)))

topologies['skipping_2'] = tuple(zip((0,4,8),
                 (1,5,9)))

topologies["zero"] = tuple(zip((0,0,0),
                 (1,1,1)))


os.rename("fountain/00010.png","fountain/0010.png")
os.rename("fountain/00011.png","fountain/0011.png")
images= sorted(glob.glob("fountain/*.png"))

print(images)

images_cv = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in images]
plt.imshow(images_cv[9])
plt.show()

def main(K, images_cv, topology):
    xyz_global_array = [None]*len(topology)
    for pair_index, (left_index,right_index) in enumerate(topology):
        print(pair_index)
        img1 = images_cv[left_index]
        img2 = images_cv[right_index]

        # 1. Feature Matching
        kp1,desc1,kp2,desc2,matches=GetImageMatches(img1,img2)
        img1pts,img2pts,img1idx,img2idx=GetAlignedMatches(kp1,desc1,kp2,desc2,matches)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    flags = 2)

        plt.imshow(cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params))
        plt.show()

        #2. Fundamental
        F, mask = cv2.findFundamentalMat(img1pts, img2pts, method=cv2.FM_7POINT)
        mask=mask.astype(bool).flatten()

        #2.2 Inliers // Optional
        img1pts = img1pts[mask==True]
        img2pts = img2pts[mask==True]
        mask = len(img1pts) * [True] ### We need the match matrix to be the same size of the number of points

        #3. Essential
        E = K.T.dot(F.dot(K))
        
        #4. R, T
        pts_rec, r_rec, t_rec, mask_rec = cv2.recoverPose(E, img1pts, img2pts)

        #5. Triangulate
        pts3d = GetTriangulatedPts(img1pts[mask],img2pts[mask],K,r_rec,t_rec)
        
        #6. Add to Global Points
        xyz_global_array[pair_index] = pts3d
        
    return xyz_global_array

full_pts3d = main(K, images_cv, topologies["overlapping"])

for idx, points in enumerate(full_pts3d):
    pts2ply(full_pts3d[idx], "out_1_{}.ply".format(idx))

xyz = np.vstack(full_pts3d)
pts2ply(xyz, "out_full.ply")
