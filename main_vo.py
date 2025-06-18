#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""
import sys

import numpy as np
import cv2
import os
import math
import time 
import platform 

from config import Config

from visual_odometry import VisualOdometryEducational
from visual_odometry_rgbd import VisualOdometryRgbd, VisualOdometryRgbdTensor
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import DatasetType, SensorType
import argparse
from mplot_thread import Mplot2d, Mplot3d
from qplot_thread import Qplot2d

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from utils_sys import Printer
from rerun_interface import Rerun

from test_config import feature_tracker_names_str
feature_tracker_names = {

    1: FeatureTrackerConfigs.LK_SHI_TOMASI,  # not in slam
    2: FeatureTrackerConfigs.LK_FAST, # not in slam
    3: FeatureTrackerConfigs.SHI_TOMASI_ORB, # very bad shoudl leave out
    4: FeatureTrackerConfigs.SHI_TOMASI_FREAK, # very bad should be skipped
    5: FeatureTrackerConfigs.FAST_ORB,
    6: FeatureTrackerConfigs.FAST_FREAK,
    7: FeatureTrackerConfigs.BRISK,
    8: FeatureTrackerConfigs.BRISK_TFEAT,
    9: FeatureTrackerConfigs.ORB,
    10: FeatureTrackerConfigs.ORB2,
    11: FeatureTrackerConfigs.KAZE,
    12: FeatureTrackerConfigs.AKAZE,
    13: FeatureTrackerConfigs.SIFT,
    14: FeatureTrackerConfigs.ROOT_SIFT,
    15: FeatureTrackerConfigs.SURF,
    16: FeatureTrackerConfigs.SUPERPOINT,
    17: FeatureTrackerConfigs.XFEAT,
    18: FeatureTrackerConfigs.XFEAT_XFEAT,
    19: FeatureTrackerConfigs.XFEAT_LIGHTGLUE,
    20: FeatureTrackerConfigs.LIGHTGLUE,
    21: FeatureTrackerConfigs.LIGHTGLUE_DISK,
    22: FeatureTrackerConfigs.LIGHTGLUE_ALIKED,
    23: FeatureTrackerConfigs.LIGHTGLUESIFT,
    24: FeatureTrackerConfigs.DELF,
    25: FeatureTrackerConfigs.D2NET,
    26: FeatureTrackerConfigs.R2D2,
    27: FeatureTrackerConfigs.LFNET,
    28: FeatureTrackerConfigs.CONTEXTDESC,
    29: FeatureTrackerConfigs.KEYNET,
    30: FeatureTrackerConfigs.DISK,
    31: FeatureTrackerConfigs.ALIKED,
    32: FeatureTrackerConfigs.KEYNETAFFNETHARDNET,
    33: FeatureTrackerConfigs.ORB2_FREAK,
    34: FeatureTrackerConfigs.ORB2_BEBLID,
    35: FeatureTrackerConfigs.ORB2_HARDNET,
    36: FeatureTrackerConfigs.ORB2_SOSNET,
    37: FeatureTrackerConfigs.ORB2_L2NET,
    38: FeatureTrackerConfigs.LOFTR,
    39: FeatureTrackerConfigs.MAST3R_MATCHER
}






kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + '/results'


kUseRerun = False
# check rerun does not have issues 
if kUseRerun and not Rerun.is_ok():
    kUseRerun = False
    
"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = False
if platform.system() == 'Darwin':
    kUsePangolin = True # Under mac force pangolin to be used since Mplot3d() has some reliability issues                
if kUsePangolin:
    from viewer3D import Viewer3D

kUseQplot2d = False
if platform.system() == 'Darwin':
    kUseQplot2d = True # Under mac force the usage of Qtplot2d: It is smoother 

def factory_plot2d(*args,**kwargs):
    if kUseRerun:
        return None
    if kUseQplot2d:
        return Qplot2d(*args,**kwargs)
    else:
        return Mplot2d(*args,**kwargs)


from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config_path', type=str, default=None,
    #                     help='Optional path for custom configuration file')
    # parser.add_argument('--no_output_date', action='store_true', help='Do not append date to output directory')
    # parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--tracker_config', type=int, help='Specify the tracker configuration as a number')
    # parser.add_argument('--save_path', type=str, help='trajectory_save_path')
    parser.add_argument('--sigma_level0',type=int,help= "Default one is 1 or 0")

    parser.add_argument('--headless',action='store_true', help='no visualization ')
    parser.add_argument('--num_levels',type = int)
    parser.add_argument('--match_ratio_test', type=float)
    parser.add_argument('--scale_factor',type =float)

    args = parser.parse_args()
    path_res = "results_VO"
    if not os.path.exists(path_res):
        os.makedirs(path_res)

    trajectory_file_save_path=  os.path.join(path_res,feature_tracker_names_str[args.tracker_config])




    config = Config()
    
    dataset = dataset_factory(config)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)

    num_features=2000  # how many features do you want to detect and track?
    if config.num_features_to_extract > 0:  # override the number of features to extract if we set something in the settings file
        num_features = config.num_features_to_extract
        
    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR
    tracker_config = feature_tracker_names[args.tracker_config]

    tracker_config['num_features'] = num_features
    tracker_config["sigma_level0"]= args.sigma_level0
    if  "num_levels"  in tracker_config.keys():
        tracker_config["num_levels"] =args.num_levels
    if "scale_factor" in tracker_config.keys():
        tracker_config["scale_factor"] = args.scale_factor
    if "match_ratio_test" in tracker_config.keys():
        tracker_config['match_ratio_test']= args.match_ratio_test
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object 
    if dataset.sensor_type == SensorType.RGBD:
        vo = VisualOdometryRgbdTensor(cam, groundtruth)  # only for RGBD
        Printer.green('Using VisualOdometryRgbdTensor')
    else:
        vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)
        Printer.green('Using VisualOdometryEducational')
    time.sleep(1) # time to read the message

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1

    plt3d = None
    
    viewer3D = None 

    is_draw_3d = True

    is_draw_with_rerun = kUseRerun
    if is_draw_with_rerun:
        Rerun.init_vo()
    else: 
        if kUsePangolin:
            viewer3D = Viewer3D(scale=dataset.scale_viewer_3d*10)
        else:
            plt3d = Mplot3d(title='3D trajectory')

    is_draw_err = True 
    err_plt = factory_plot2d(xlabel='img id', ylabel='m',title='error')
    
    is_draw_matched_points = True 
    matched_points_plt = factory_plot2d(xlabel='img id', ylabel='# matches',title='# matches')
    # if (arsg.headless):
    #     is_draw_3d = False
    #     is_draw_traj_img = False
    #     is_draw_err = False
    #     is_draw_matched_points =False

    img_id = 0
    while True:
        
        img = None

        if dataset.isOk():
            timestamp = dataset.getTimestamp()          # get current timestamp 
            img = dataset.getImageColor(img_id)
            depth = dataset.getDepth(img_id)

        if img is not None:

            vo.track(img, depth, img_id, timestamp)  # main VO function 

            if(len(vo.traj3d_est)>1):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                gt_x, gt_y, gt_z = vo.traj3d_gt[-1]

                if is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                    draw_gt_x, draw_gt_y = int(draw_scale*gt_x) + half_traj_img_size, half_traj_img_size - int(draw_scale*gt_z)
                    cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                    cv2.circle(traj_img, (draw_gt_x, draw_gt_y), 1,(0, 0, 255), 1)  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                    # show 		

                    if is_draw_with_rerun:
                        Rerun.log_img_seq('trajectory_img/2d', img_id, traj_img)
                    else:
                        cv2.imshow('Trajectory', traj_img)


                if is_draw_with_rerun:                                        
                    Rerun.log_2d_seq_scalar('trajectory_error/err_x', img_id, math.fabs(gt_x-x))
                    Rerun.log_2d_seq_scalar('trajectory_error/err_y', img_id, math.fabs(gt_y-y))
                    Rerun.log_2d_seq_scalar('trajectory_error/err_z', img_id, math.fabs(gt_z-z))
                    
                    Rerun.log_2d_seq_scalar('trajectory_stats/num_matches', img_id, vo.num_matched_kps)
                    Rerun.log_2d_seq_scalar('trajectory_stats/num_inliers', img_id, vo.num_inliers)
                    
                    Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, None, cam, vo.poses[-1])
                    Rerun.log_3d_trajectory(img_id, vo.traj3d_est, 'estimated', color=[0,0,255])
                    Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, 'ground_truth', color=[255,0,0])     
                else:
                    if is_draw_3d:           # draw 3d trajectory 
                        if kUsePangolin:
                            viewer3D.draw_vo(vo)   
                        else:
                            plt3d.draw(vo.traj3d_gt,'ground truth',color='r',marker='.')
                            plt3d.draw(vo.traj3d_est,'estimated',color='g',marker='.')

                    if is_draw_err:         # draw error signals 
                        errx = [img_id, math.fabs(gt_x-x)]
                        erry = [img_id, math.fabs(gt_y-y)]
                        errz = [img_id, math.fabs(gt_z-z)] 
                        err_plt.draw(errx,'err_x',color='g')
                        err_plt.draw(erry,'err_y',color='b')
                        err_plt.draw(errz,'err_z',color='r')

                    if is_draw_matched_points:
                        matched_kps_signal = [img_id, vo.num_matched_kps]
                        inliers_signal = [img_id, vo.num_inliers]                    
                        matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
                        matched_points_plt.draw(inliers_signal,'# inliers',color='g')                                                     
                    
            # draw camera image 
            if not is_draw_with_rerun:
                cv2.imshow('Camera', vo.draw_img)				

        else: 
            time.sleep(0.1) 
                
        # get keys 
        key = matched_points_plt.get_key() if matched_points_plt is not None else None
        if key == '' or key is None:
            key = err_plt.get_key() if err_plt is not None else None
        if key == '' or key is None:
            key = plt3d.get_key() if plt3d is not None else None
            
        # press 'q' to exit!
        key_cv = cv2.waitKey(1) & 0xFF
        if key == 'q' or (key_cv == ord('q')):            
            break
        if viewer3D and viewer3D.is_closed():
            break
        img_id += 1
        if(img is None):
            break


    #print('press a key in order to exit...')
    #cv2.waitKey(0)

    if is_draw_traj_img:
        if not os.path.exists(kResultsFolder):
            os.makedirs(kResultsFolder, exist_ok=True)
        print(f'saving {kResultsFolder}/map.png')
        cv2.imwrite(f'{kResultsFolder}/map.png', traj_img)
    if is_draw_3d:
        if not kUsePangolin:
            plt3d.quit()
        else: 
            viewer3D.quit()
    if is_draw_err:
        err_plt.quit()
    if is_draw_matched_points is not None:
        matched_points_plt.quit()
    np.save(f'{trajectory_file_save_path}.npy', vo.traj3d_est)
    sys.exit()
    cv2.destroyAllWindows()

