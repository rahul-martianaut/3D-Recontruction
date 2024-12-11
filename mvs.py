import numpy
import pathlib
from pathlib import Path
from time import time
import os
import inspect
pwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
from mvs.load_ply import save_ply
from mvs.load_camera_info import load_intrinsics, load_extrinsics
import cv2
import glob
import matplotlib.pyplot as plt
import collections
from mvs.load_camera_info import load_all_camera_parameters

 
topologies = collections.OrderedDict()
topologies['360'] = tuple(zip((0,1,2,3,4,5,6,7,8,9,10,11),
                          (1,2,3,4,5,6,7,8,9,10,11,0)))

topologies['overlapping'] = tuple(zip((0,1,2,3,4,5,6,7,8,9,10),
                          (1,2,3,4,5,6,7,8,9,10,11)))

topologies['adjacent'] = tuple(zip((0,2,4,6,8,10),
                     (1,3,5,7,9,11)))
topologies['skipping_1'] = tuple(zip((0,3,6,9),
                 (1,4,7,10)))
topologies['skipping_2'] = tuple(zip((0,4,8),
                 (1,5,9)))

for pair_index, (left_index,right_index) in enumerate(topologies["360"]):
    print(left_index, right_index)



def get_camera_parameters(options):
    flags=options['StereoRectify']['flags']
    distortion_coefficients = (0.0,0.0,0.0,0.0,0.0)
    left_distortion_coefficients = distortion_coefficients
    right_distortion_coefficients = distortion_coefficients
    imageSize = options['StereoRectify']['imageSize'] # w,h
    newImageSize = options['StereoRectify']['newImageSize']
    alpha = options['StereoRectify']['alpha']
    return left_distortion_coefficients, right_distortion_coefficients, imageSize, newImageSize, alpha


def calibrate_and_rectify(options, left_K, right_K, left_R, right_R, left_T, right_T):
    # Get the parameters
    left_distortion_coefficients, right_distortion_coefficients, imageSize, newImageSize, alpha = get_camera_parameters(options)
           
    # Stereo Rectify
    R_intercamera = numpy.dot(right_R, left_R.T) # R * T
    T_intercamera = right_T - numpy.dot(R_intercamera, left_T) # translation and rotation keeping the first one as baseline

    left_R_rectified, right_R_rectified, P1_rect, P2_rect, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1 = left_K, distCoeffs1 = left_distortion_coefficients,
        cameraMatrix2 = right_K, distCoeffs2 = right_distortion_coefficients,
        imageSize=imageSize, 
        newImageSize=newImageSize,
        R=R_intercamera, T=T_intercamera,
        flags=options['StereoRectify']['flags'] , alpha=alpha)

    # Back to Global
    R2,T2 = left_R, left_T # perspective is from left image.
    R3,T3 = R2.T,numpy.dot(-R2.T,T2) # Invert direction of transformation to map camera to world.
    R_left_rectified_to_global = numpy.dot(R3,left_R_rectified.T)
    T_left_rectified_to_global = T3
    extrinsics_left_rectified_to_global = R_left_rectified_to_global.astype(numpy.float32), T_left_rectified_to_global.astype(numpy.float32)

    # Create rectification maps
    rectification_map_type = cv2.CV_16SC2
    left_maps = cv2.initUndistortRectifyMap(left_K,
                                            left_distortion_coefficients,
                                            left_R_rectified,
                                            P1_rect,
                                            size=newImageSize,
                                            m1type=rectification_map_type)
    
    right_maps = cv2.initUndistortRectifyMap(right_K,
                                            right_distortion_coefficients,
                                            right_R_rectified,
                                            P2_rect,
                                            size=newImageSize,
                                            m1type=rectification_map_type)

    return Q, extrinsics_left_rectified_to_global, left_maps, right_maps

def get_disparity_matcher(options):
    # Instantiate the matchers; they may do something slow internally...
    if 'StereoBM' in options:
        # Perform stereo matching using normal block matching
        numDisparities = options['StereoMatcher']['NumDisparities']
        blockSize = options['StereoMatcher']['BlockSize']
        matcher = cv2.StereoBM_create(numDisparities=numDisparities,blockSize=blockSize)
        setterOptions = {}
        setterOptions.update(options['StereoMatcher'])
        setterOptions.update(options['StereoBM'])
        for key,value in setterOptions.items():
            setter = eval('matcher.set'+key) # Returns the setter function
            setter(value) # Calls the setter function.
    
    elif 'StereoSGBM' in options:
        # Perform stereo matching using SGBM
        minDisparity = options['StereoMatcher']['MinDisparity']
        numDisparities = options['StereoMatcher']['NumDisparities']
        blockSize = options['StereoMatcher']['BlockSize']
        matcher = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                    numDisparities=numDisparities,
                                    blockSize=blockSize)
        setterOptions = {}
        setterOptions.update(options['StereoMatcher'])
        setterOptions.update(options['StereoSGBM'])
        for key,value in setterOptions.items():
            setter = eval('matcher.set'+key) # Returns the setter function
            setter(value) # Calls the setter function.
    else:
        assert False, "Couldn't determine the matcher type from passed options!"

    return matcher



class OpenCVStereoMatcher():
    def __init__(self,options=None,calibration_path=None):
        self.options = options
        self.num_cameras = options['CameraArray']['num_cameras']
        self.topology = options['CameraArray']['topology']
        self.all_camera_parameters = load_all_camera_parameters(calibration_path)

        self.left_maps_array = []
        self.right_maps_array = []
        self.Q_array = []
        self.extrinsics_left_rectified_to_global_array = []

        for pair_index, (left_index,right_index) in enumerate(topologies[self.topology]):
            ## 1 — Get R, T, W, H for each camera
            left_K, left_R, left_T, left_width, left_height = [self.all_camera_parameters[left_index][key] for key in ('camera_matrix','R','T','image_width','image_height')]
            right_K, right_R, right_T, right_width, right_height = [self.all_camera_parameters[right_index][key] for key in ('camera_matrix','R','T','image_width','image_height')]
            h,w = left_height, left_width

            # 2 — Stereo Calibrate & Rectify
            Q, extrinsics_left_rectified_to_global, left_maps, right_maps = calibrate_and_rectify(options, left_K, right_K,
                                                                                                  left_R, right_R,
                                                                                                  left_T, right_T)
            self.Q_array.append(Q)
            self.extrinsics_left_rectified_to_global_array.append(extrinsics_left_rectified_to_global)
            self.left_maps_array.append(left_maps)
            self.right_maps_array.append(right_maps)

            # 3 — Get Matcher
            self.matcher = get_disparity_matcher(options)

    def load_images(self,imagesPath):
        # Load a set of images from disk. Doesn't do processing yet.
        imagesPath = imagesPath.resolve()

        # Load the undistorted images off of disk
        print('Loading the images off of disk...')
        num_cameras = len(list(imagesPath.glob('*.png')))
        assert self.num_cameras == num_cameras, 'Mismatch in the number of available images!'
        images = []
        for i in range(num_cameras):
            fileName = 'image_camera%02i.png' % (i + 1)
            filePath = imagesPath / fileName
            print('Loading image',filePath)
            colorImage = cv2.imread(str(filePath))
            grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
            images.append(grayImage)
            expected_parameters = self.all_camera_parameters[i]
            w,h = expected_parameters['image_width'], expected_parameters['image_height']
            assert grayImage.shape == (h,w), 'Mismatch in image sizes!'
        self.images = images

    def run(self):
        assert self.all_camera_parameters is not None, 'Camera parameters not loaded yet; You should run load_all_camera_parameters first!'
        xyz_global_array = [None]*len(topologies[self.topology])

        def run_pair(pair_idx, left_idx, right_idx):
            # Load the proper images and rectification maps
            left_img, right_img = self.images[left_idx], self.images[right_idx]
            left_maps = self.left_maps_array[pair_idx]
            right_maps = self.right_maps_array[pair_idx]

            # Rectify
            remap_interpolation = self.options['Remap']['interpolation']
            left_image_rectified = cv2.remap(left_img, left_maps[0],left_maps[1], remap_interpolation)
            right_image_rectified = cv2.remap(right_img, right_maps[0], right_maps[1], remap_interpolation)

            # Load & Find Disparity
            disparity_image = self.matcher.compute(left_image_rectified, right_image_rectified)
            
            if disparity_image.dtype == numpy.int16:
                disparity_image = disparity_image.astype(numpy.float32)
                disparity_image /= 16
            
            plt.imshow(disparity_image)
            plt.show()

            # Reproject 3D
            Q = self.Q_array[pair_idx]
            threedeeimage = cv2.reprojectImageTo3D(disparity_image, Q, handleMissingValues=True,ddepth=cv2.CV_32F)
            threedeeimage = numpy.array(threedeeimage)

            # Postprocess
            xyz = threedeeimage.reshape((-1,3)) # x,y,z now in three columns, in left rectified camera coordinates
            z = xyz[:,2]
            goodz = z < 9999.0 
            xyz_filtered = xyz[goodz,:]
            
            # Global Coordinates
            R_left_rectified_to_global, T_left_rectified_to_global = self.extrinsics_left_rectified_to_global_array[pair_idx]
            xyz_global = numpy.dot(xyz_filtered, R_left_rectified_to_global.T) + T_left_rectified_to_global.T 

            # Save PLY
            save_ply(xyz_global, os.path.join('output/rock','pair_'+str(left_index)+'_'+str(right_index)+'.ply'))
            xyz_global_array[pair_index] = xyz_global

        for pair_index, (left_index,right_index) in enumerate(topologies[self.topology]):
            run_pair(pair_index, left_index, right_index)

        xyz = numpy.vstack(xyz_global_array)
        return xyz




if __name__ == "__main__":

    

    scenario = "rock" 
    images = sorted(glob.glob("mvs/data/"+scenario+"/undistorted/*.png")) 
    print("images ", len(images))
    imagesPath = Path('mvs/data/'+scenario+'/undistorted')
    workDirectory=Path('.') 

    if scenario == "rock":
        images_cv = [cv2.rotate(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), cv2.ROTATE_180) for img in images]
        #images_cv = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in images]

    elif scenario=="temple":
        #images_cv = [cv2.rotate(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_COUNTERCLOCKWISE) for img in images]
        images_cv = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in images]

    h, w, d = images_cv[3].shape  

    StereoMatcherOptions = {'MinDisparity': -64, # Influences MAX depth
                        'NumDisparities': 256, # Influences MIN depth
                        'BlockSize': 21,
                        'SpeckleWindowSize': 0, # Must be strictly positive to turn on speckle post-filter.
                        'SpeckleRange': 0, # Must be >= 0 to enable speckle post-filter
                        'Disp12MaxDiff': 0}
                        
    StereoBMOptions = {
            'PreFilterType': (cv2.StereoBM_PREFILTER_NORMALIZED_RESPONSE, cv2.StereoBM_PREFILTER_XSOBEL)[0],
                    'PreFilterSize': 5, # preFilterSize must be odd and be within 5..255
                    'PreFilterCap': 63, # preFilterCap must be within 1..63. Used to truncate pixel values
                    'TextureThreshold': 10,
                    'UniquenessRatio': 10,
                    }

    StereoSGBMOptions = {'PreFilterCap': 0,
                        'UniquenessRatio': 0,
                        'P1': 16*21*21, # "Depth Change Cost in Ensenso terminology"
                        'P2': 16*21*21, # "Depth Step Cost in Ensenso terminology"
                        'Mode': (cv2.StereoSGBM_MODE_SGBM, cv2.StereoSGBM_MODE_HH,
                                cv2.StereoSGBM_MODE_SGBM_3WAY)[1]}

    StereoRectifyOptions = {'imageSize':(w,h),
                            'flags':(0,cv2.CALIB_ZERO_DISPARITY)[0], # TODO explore other flags
                            'newImageSize':(w,h),
                            'alpha':0.5}

    RemapOptions = {'interpolation':cv2.INTER_LINEAR}

    CameraArrayOptions = {'channels':1,'num_cameras':12,'topology':'overlapping'} #topology=skipping_1, skipping_2, adjacent, overlapping, 360

    FinalOptions = {'StereoRectify':StereoRectifyOptions,
            'StereoMatcher':StereoMatcherOptions,
            'StereoBM':StereoBMOptions,#change to "StereoSGBM" = StereoSGBMOptions if needed
            'CameraArray':CameraArrayOptions,
            'Remap':RemapOptions}

    opencv_matcher = OpenCVStereoMatcher(options=FinalOptions, calibration_path=imagesPath)
    opencv_matcher.load_images(imagesPath)
    xyz = opencv_matcher.run()

    save_ply(xyz, os.path.join("output/rock","therock_new.ply"))


