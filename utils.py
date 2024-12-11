import numpy as np
import cv2


def GetImageMatches(img1,img2):
    # sift=cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1,None)
    kp2, desc2 = sift.detectAndCompute(img2,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key = lambda x:x.distance)

    return kp1,desc1,kp2,desc2,matches

def GetAlignedMatches(kp1,desc1,kp2,desc2,matches):

    #Sorting in case matches array isn't already sorted
    matches = sorted(matches, key = lambda x:x.distance)

    #retrieving corresponding indices of keypoints (in both images) from matches.  
    img1idx = np.array([m.queryIdx for m in matches])
    img2idx = np.array([m.trainIdx for m in matches])

    #filtering out the keypoints that were NOT matched. 
    kp1_ = (np.array(kp1))[img1idx]
    kp2_ = (np.array(kp2))[img2idx]

    #retreiving the image coordinates of matched keypoints
    img1pts = np.array([kp.pt for kp in kp1_])
    img2pts = np.array([kp.pt for kp in kp2_])

    return img1pts,img2pts,img1idx,img2idx

def ComputeEpiline(pts, index, F):     
    if pts.shape[1]==2:
        #converting to homogenous coordinates if not already
        pts = cv2.convertPointsToHomogeneous(pts)[:,0,:]
        
    if index==1: 
        lines = F.dot(pts.T) 
    elif index==2: 
        lines = F.T.dot(pts.T)
    
    return lines.T

def drawlines(img1,img2,lines,pts1,pts2,drawOnly=None,linesize=7,circlesize=10):
    r,c = img1.shape[:-1]

    img1_, img2_ = np.copy(img1), np.copy(img2)

    drawOnly = lines.shape[0] if (drawOnly is None) else drawOnly

    i = 0 
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        
        img1_ = cv2.line(img1_, (x0,y0), (x1,y1), color,linesize)
        img1_ = cv2.circle(img1_,tuple(pt1.astype(int)),circlesize,color,-1)
        img2_ = cv2.circle(img2_,tuple(pt2.astype(int)),circlesize,color,-1)

        i += 1 
        
        if i > drawOnly: 
            break 

    return img1_,img2_

def GetTriangulatedPts(img1pts,img2pts,K,R,t): 
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]
    
    pts4d = cv2.triangulatePoints(np.eye(3,4),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]
    
    return pts3d

def pts2ply(pts,filename='out.ply'): 
    f = open(filename,'w')
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(pts.shape[0]))
    
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    
    f.write('end_header\n')
    
    for pt in pts: 
        f.write('{} {} {} 255 255 255\n'.format(pt[0],pt[1],pt[2]))
    f.close()