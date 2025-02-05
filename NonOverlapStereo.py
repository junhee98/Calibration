import os
import cv2
import glob
import itertools
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class NonOverlapStereoCalib:
    def __init__(self, left_camera, right_camera):
        # Cameras
        self.left_camera = left_camera
        self.right_camera = right_camera

        if self.left_camera.image_shape != self.right_camera.image_shape:
            raise ValueError("Image shapes of left and right cameras are different!")
        else:
            self.image_shape = self.left_camera.image_shape

        # Common points
        self.common_3d_points = []
        self.common_left_corner_points = []
        self.common_right_corner_points = []

        # Results
        self.output_path = self.left_camera.output_path

        # Load intrinsic parameters
        self.K_1 = self.left_camera.K
        self.d_1 = self.left_camera.d

        self.K_2 = self.right_camera.K
        self.d_2 = self.right_camera.d

        self.rvecs_left = []
        self.tvecs_left = []

        self.rvecs_right = []
        self.tvecs_right = []

        self.R = None
        self.T = None

        self.map1_x = None
        self.map1_y = None
        self.map2_x = None
        self.map2_y = None

        self.roi_1 = None 
        self.roi_2 = None

    def redetectCharucoCornersAndIds(self, image, board):
        """
        Redetect Charuco corners and ids
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, board.aruco_dict)

        # Refine detected markers
        if ids is not None:
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(gray, board.board, corners, ids, None, None, None)

            # Detect charuco corners
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board.board)

            if charuco_ids is not None:
                if len(charuco_corners) >= 6: # at least 6 corners are required for DLT algorithm in calibration
                    # Find object points for calcuate reproection error
                    objpoints = []
                    for charuco_id in charuco_ids.flatten():
                        objpoints.append(board.charuco_3d_points[charuco_id])
                    return "true", charuco_corners, charuco_ids, np.array(objpoints, dtype=np.float32)
                else:
                    return "few_corners", None, None, None
            else:
                return "non_corners", None, None, None

        return "non_markers", None, None, None

    def CheckCommonPoints(self):
        """
        Check common points between left and right cameras for analysis non-overlapping stereo calibration
        """
        # check total length of frames
        left_images = sorted(glob.glob(os.path.join(self.left_camera.root_path, f'{self.left_camera.image_prefix}', f'*.{self.left_camera.image_format}')))
        right_images = sorted(glob.glob(os.path.join(self.right_camera.root_path, f'{self.right_camera.image_prefix}', f'*.{self.right_camera.image_format}')))

        if len(left_images) != len(right_images):
            raise ValueError("The number of images in left and right cameras are different!")
        
        # Find same idx for left and right images -- these used for analysis non-overlapping stereo calibration
        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['state'] != right_frame['state']: # check the state of the frames
                if left_frame['state'] == 'true' and right_frame['state'] == 'non_corners': # few overlap points is left camera
                    print(f"Frame {left_frame['filename']} is detected in the left camera but not in the right camera. -- Find common points for Analysis!")

                    # detect corners in the right camera with left camera's board
                    state, charuco_corners, charuco_ids, objpoints = self.redetectCharucoCornersAndIds(right_frame['image'], self.left_camera.board)

                    right_frame['charuco_corners'] = charuco_corners
                    right_frame['charuco_ids'] = charuco_ids

                    # Find common points
                    common_ids = np.intersect1d(left_frame['charuco_ids'].flatten(), charuco_ids.flatten())

                    if len(common_ids) == 0:
                        print("No common corners detected between the two cameras for this frame. Skipping frame.")
                        continue

                    if len(common_ids) >= 6: # at least 6 common points for stereo calibration
                        obj_points = self.left_camera.board.charuco_3d_points[common_ids]
                        img_pts_left = np.array([left_frame['charuco_corners'][left_frame['charuco_ids'].flatten() == id][0] for id in common_ids])
                        img_pts_right = np.array([charuco_corners[charuco_ids.flatten() == id][0] for id in common_ids])

                        self.common_3d_points.append(obj_points)
                        self.common_left_corner_points.append(img_pts_left)
                        self.common_right_corner_points.append(img_pts_right)
                    else:
                        print("Few common corners detected between the two cameras for this frame. Skipping frame.")
                        continue

                elif right_frame['state'] == 'true' and left_frame['state'] == 'non_corners': # few overlap points is right camera
                    print(f"Frame {right_frame['filename']} is detected in the right camera but not in the left camera. -- Find common points for Analysis!")

                    # detect corners in the right camera with left camera's board
                    state, charuco_corners, charuco_ids, objpoints = self.redetectCharucoCornersAndIds(left_frame['image'], self.right_camera.board)

                    left_frame['charuco_corners'] = charuco_corners
                    left_frame['charuco_ids'] = charuco_ids

                    # Find common points
                    common_ids = np.intersect1d(charuco_ids.flatten(), right_frame['charuco_ids'].flatten())

                    if len(common_ids) == 0:
                        print("No common corners detected between the two cameras for this frame. Skipping frame.")
                        continue

                    if len(common_ids) >= 6: # at least 6 common points for stereo calibration
                        obj_points = self.right_camera.board.charuco_3d_points[common_ids]
                        img_pts_left = np.array([charuco_corners[charuco_ids.flatten() == id][0] for id in common_ids])
                        img_pts_right = np.array([right_frame['charuco_corners'][right_frame['charuco_ids'].flatten() == id][0] for id in common_ids])

                        self.common_3d_points.append(obj_points)
                        self.common_left_corner_points.append(img_pts_left)
                        self.common_right_corner_points.append(img_pts_right)
                    else:
                        print("Few common corners detected between the two cameras for this frame. Skipping frame.")
                        continue
    
    def BoardPoseEstimation(self):
        """
        Board pose estimation for using handeye calibration
        """
        rvec_left_list = []
        tvec_left_list = []

        rvec_right_list = []
        tvec_right_list = []

        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['state'] == 'true' and right_frame['state'] == 'true':
                # for left case
                rvec_left = np.zeros((3, 1), dtype=np.float32) 
                tvec_left = np.zeros((3, 1), dtype=np.float32)
                _, rvec_left, tvec_left = aruco.estimatePoseCharucoBoard(
                    left_frame['charuco_corners'], left_frame['charuco_ids'], self.left_camera.board.board, self.left_camera.K, self.left_camera.d, rvec_left, tvec_left)
                rvec_left_list.append(rvec_left)
                tvec_left_list.append(tvec_left)

                # for right case
                rvec_right = np.zeros((3, 1), dtype=np.float32)
                tvec_right = np.zeros((3, 1), dtype=np.float32)
                _, rvec_right, tvec_right = aruco.estimatePoseCharucoBoard(
                    right_frame['charuco_corners'], right_frame['charuco_ids'], self.right_camera.board.board, self.right_camera.K, self.right_camera.d, rvec_right, tvec_right)
                rvec_right_list.append(rvec_right)
                tvec_right_list.append(tvec_right)

        return rvec_left_list, tvec_left_list, rvec_right_list, tvec_right_list

    def ConvertBoard2Camera(self, rvecs, tvecs, left=True):
        """
        Convert camera based board pose to board based camera pose
        """

        for i in range(len(rvecs)):
            R = cv2.Rodrigues(rvecs[i])[0]
            R_inv = R.T
            rvec_inv = cv2.Rodrigues(R_inv)[0]
            tvec_inv = -np.dot(R_inv, tvecs[i])

            if left:
                self.rvecs_left.append(rvec_inv)
                self.tvecs_left.append(tvec_inv)
            else:
                self.rvecs_right.append(rvec_inv)
                self.tvecs_right.append(tvec_inv)

    def initNonOverlapPair(self, left=True):
        """
        Initialize pair for non-overlapping stereo calibration
        """

        # Construct Initialize pair for Handeye calibration
        N, rvecs, tvecs = None, None, None
        pose_abs = []
        if left:
            N = len(self.rvecs_left)
        else:
            N = len(self.rvecs_right)
        
        combinations = list(itertools.combinations(range(N), 2))

        if left:
            rvecs = self.rvecs_left
            tvecs = self.tvecs_left
        else:
            rvecs = self.rvecs_right
            tvecs = self.tvecs_right

        for i, j in combinations:
            # pose1
            R1 = cv2.Rodrigues(rvecs[i])[0]
            T1 = tvecs[i]
            pose1 = np.eye(4)
            pose1[:3, :3] = R1
            pose1[:3, 3] = T1.T

            # pose2
            R2 = cv2.Rodrigues(rvecs[j])[0]
            T2 = tvecs[j]
            pose2 = np.eye(4)
            pose2[:3, :3] = R2
            pose2[:3, 3] = T2.T

            pose1_inv = np.linalg.inv(pose1)
            f2f = np.dot(pose1_inv, pose2)

            pose_abs.append(f2f)
        
        return pose_abs
    
    def ConstructLoop(self, pose_abs_1, pose_abs_2):
        """
        Construct loop for handeye calibration
        """

        num_poses = len(self.rvecs_right)

        r_cam_1_list, r_cam_2_list, t_cam_1_list, t_cam_2_list = [], [], [], []

        for i in range(num_poses):
            pose_cam1 = np.linalg.inv(pose_abs_1[i])
            pose_cam2 = pose_abs_2[i]
            
            R = pose_cam1[:3, :3]
            t = pose_cam1[:3, 3:]

            r_cam_1, t_cam_1 = R, t

            R = pose_cam2[:3, :3]
            t = pose_cam2[:3, 3:]

            r_cam_2, t_cam_2 = R, t

            r_1_vec = cv2.Rodrigues(r_cam_1)[0]
            r_2_vec = cv2.Rodrigues(r_cam_2)[0]

            r_cam_1_list.append(r_1_vec)
            t_cam_1_list.append(t_cam_1)
            r_cam_2_list.append(r_2_vec)
            t_cam_2_list.append(t_cam_2)
        
        return r_cam_1_list, t_cam_1_list, r_cam_2_list, t_cam_2_list

    def NonOverlapStereoCalibration(self):
        """
        Non-overlap Stereo calibration
        """

        # Pose estimation for handeye calibration
        rvecs_left, tvecs_left, rvecs_right, tvecs_right = self.BoardPoseEstimation()

        # Convert camera based board pose to board based camera pose
        self.ConvertBoard2Camera(rvecs_left, tvecs_left, left=True)
        self.ConvertBoard2Camera(rvecs_right, tvecs_right, left=False)

        # Construct initial for Handeye calibration
        pose_abs_1 = self.initNonOverlapPair(left=True)
        pose_abs_2 = self.initNonOverlapPair(left=False)

        # Construct loop for handeye calibration
        r_cam_1_list, t_cam_1_list, r_cam_2_list, t_cam_2_list = self.ConstructLoop(pose_abs_1, pose_abs_2)

        # Handeye calibration
        R_c1_c2, t_c1_c2 = cv2.calibrateHandEye(
            r_cam_1_list, 
            t_cam_1_list, 
            r_cam_2_list, 
            t_cam_2_list, 
            method=cv2.CALIB_HAND_EYE_HORAUD
        )

        # Construct R and T
        cam2_position_world = -np.dot(R_c1_c2.T, t_c1_c2)
        cam2_orientation_world = R_c1_c2.T

        self.R = cam2_orientation_world
        self.T = cam2_position_world
        
    def Rectification(self):
        """
        Rectification for small overalpping stereo cameras
        """

        # Conduct stereo rectification
        R_1, R_2, P_1, P_2, Q, self.roi_1, self.roi_2 = cv2.stereoRectify(
            cameraMatrix1 = self.K_1,
            distCoeffs1 = self.d_1,
            cameraMatrix2 = self.K_2,
            distCoeffs2 = self.d_2,
            imageSize = self.image_shape,
            R = self.R,
            T = self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # Undistort and rectify images for left camera
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(
            cameraMatrix = self.K_1,
            distCoeffs = self.d_1,
            R = R_1,
            newCameraMatrix = P_1,
            size = self.image_shape,
            m1type=cv2.CV_32FC1
        )

        # Undistort and rectify images for right camera
        self.map2_x, self.map2_y = cv2.initUndistortRectifyMap(
            cameraMatrix = self.K_2,
            distCoeffs = self.d_2,
            R = R_2,
            newCameraMatrix = P_2,
            size = self.image_shape,
            m1type=cv2.CV_32FC1
        )

    def visualization(self):
        """
        Visualization
        """

        # Visualization detected corners in images
        if not os.path.exists(os.path.join(self.output_path, self.left_camera.image_prefix, 'Rectified_NonOverlap')):
            os.makedirs(os.path.join(self.output_path, self.left_camera.image_prefix, 'Rectified_NonOverlap'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, self.right_camera.image_prefix, 'Rectified_NonOverlap')):
            os.makedirs(os.path.join(self.output_path, self.right_camera.image_prefix, 'Rectified_NonOverlap'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, 'Rectified_NonOverlap')):
            os.makedirs(os.path.join(self.output_path, 'Rectified_NonOverlap'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, 'Projection_NonOverlap')):
            os.makedirs(os.path.join(self.output_path, 'Projection_NonOverlap'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, 'StereoAnalysis_NonOverlap')):
            os.makedirs(os.path.join(self.output_path, 'StereoAnalysis_NonOverlap'), exist_ok=True)

        rectified_y_errors = []
        all_y_errors = []

        projection_errors = []
        all_projection_errors = []

        # Visualization Rectified images
        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['state'] == 'true' and right_frame['state'] == 'true':
                # Rectify images
                left_rectified = cv2.remap(left_frame['image'].copy(), self.map1_x, self.map1_y, cv2.INTER_LINEAR)
                right_rectified = cv2.remap(right_frame['image'].copy(), self.map2_x, self.map2_y, cv2.INTER_LINEAR)

                # Save rectified images ( Each images )
                cv2.imwrite(os.path.join(self.output_path, self.left_camera.image_prefix, 'Rectified_NonOverlap', left_frame['filename']), left_rectified)
                cv2.imwrite(os.path.join(self.output_path, self.right_camera.image_prefix, 'Rectified_NonOverlap', right_frame['filename']), right_rectified)

        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['state'] != right_frame['state'] and (left_frame['state'] == 'non_corners' or right_frame['state'] == 'non_corners'):
                board = None
                if left_frame['state'] == 'true':
                    board = self.left_camera.board
                elif right_frame['state'] == 'true':
                    board = self.right_camera.board

                # Rectify images
                left_rectified = cv2.remap(left_frame['image'].copy(), self.map1_x, self.map1_y, cv2.INTER_LINEAR)
                right_rectified = cv2.remap(right_frame['image'].copy(), self.map2_x, self.map2_y, cv2.INTER_LINEAR)
                
                # Save rectified images ( Combined )
                combined = np.hstack((left_rectified, right_rectified))
                for line in range(0, combined.shape[0], 50): # draw horizontal lines with 50 pixel intervals
                    cv2.line(combined, (0, line), (combined.shape[1], line), (0, 255, 0), 1)
        
                cv2.imwrite(os.path.join(self.output_path, 'Rectified_NonOverlap', left_frame['filename']), combined)

                # Calculate Rectification Error
                gray_left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
                gray_right_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

                corners_left_rectified, ids_left_rectified, _ = aruco.detectMarkers(gray_left_rectified, board.aruco_dict)
                corners_right_rectified, ids_right_rectified, _ = aruco.detectMarkers(gray_right_rectified, board.aruco_dict)

                corners_left_rectified, ids_left_rectified, _, _ = aruco.refineDetectedMarkers(
                    gray_left_rectified, 
                    board.board, 
                    corners_left_rectified, 
                    ids_left_rectified, 
                    None, 
                    None, 
                    None
                )

                corners_right_rectified, ids_right_rectified, _, _ = aruco.refineDetectedMarkers(
                    gray_right_rectified, 
                    board.board, 
                    corners_right_rectified, 
                    ids_right_rectified, 
                    None, 
                    None, 
                    None
                )

                if ids_left_rectified is not None and ids_right_rectified is not None:
                    _, charuco_corners_left_rectified, charuco_ids_left_rectified = aruco.interpolateCornersCharuco(
                        corners_left_rectified, 
                        ids_left_rectified, 
                        gray_left_rectified, 
                        board.board
                    )

                    _, charuco_corners_right_rectified, charuco_ids_right_rectified = aruco.interpolateCornersCharuco(
                        corners_right_rectified, 
                        ids_right_rectified, 
                        gray_right_rectified, 
                        board.board
                    )

                    if charuco_ids_left_rectified is not None and charuco_ids_right_rectified is not None:
                        left_ids_rectified = charuco_ids_left_rectified.flatten()
                        right_ids_rectified = charuco_ids_right_rectified.flatten()
                        common_ids_rectified = np.intersect1d(left_ids_rectified, right_ids_rectified)

                        if len(common_ids_rectified) > 0:
                            common_left_corners_rectified = []
                            common_right_corners_rectified = []

                            for common_id in common_ids_rectified:
                                idx_left_rectified = np.where(left_ids_rectified == common_id)[0][0]
                                idx_right_rectified = np.where(right_ids_rectified == common_id)[0][0]

                                common_left_corners_rectified.append(charuco_corners_left_rectified[idx_left_rectified])
                                common_right_corners_rectified.append(charuco_corners_right_rectified[idx_right_rectified])

                            common_left_corners_rectified = np.array(common_left_corners_rectified).reshape(-1, 2)
                            common_right_corners_rectified = np.array(common_right_corners_rectified).reshape(-1, 2)

                            y_errors = np.abs(common_left_corners_rectified[:, 1] - common_right_corners_rectified[:, 1])
                            rectified_y_errors.append(np.mean(y_errors)) # image level
                            all_y_errors.extend(y_errors) # point level

        if len(rectified_y_errors) != 0:
            mean_y_error = np.mean(all_y_errors)

            # Visualization Rectification Y-Error per Image
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(rectified_y_errors)), rectified_y_errors, color='blue', alpha=0.7)
            plt.axhline(mean_y_error, color='r', linestyle='dashed', linewidth=2, label="Mean Y-Error")
            plt.title(f"Rectification Y-Error per Image ( Mean: {mean_y_error:.4f} pixels )")
            plt.xlabel("Image index")
            plt.ylabel("Y-Error (pixels)")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.output_path, 'StereoAnalysis_NonOverlap/rectification_y_errors.png'))

            print("Visualizing rectified images and rectification errors done!")

        # Projection Common points in left image to right image -- left cam에서 사용한 board를 이 과정을 위해서 사용!
        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['state'] != right_frame['state'] and (left_frame['state'] == 'non_corners' or right_frame['state'] == 'non_corners'):
                # Find common points
                common_ids = np.intersect1d(left_frame['charuco_ids'].flatten(), right_frame['charuco_ids'].flatten())

                if len(common_ids) > 0:
                    common_objecct_points = []
                    common_left_corners = []
                    common_right_corners = []

                    for common_id in common_ids:
                        obj_points = self.left_camera.board.charuco_3d_points[common_id]
                        idx_left = np.where(left_frame['charuco_ids'].flatten() == common_id)[0][0]
                        idx_right = np.where(right_frame['charuco_ids'].flatten() == common_id)[0][0]

                        common_objecct_points.append(obj_points)
                        common_left_corners.append(left_frame['charuco_corners'][idx_left])
                        common_right_corners.append(right_frame['charuco_corners'][idx_right])

                    common_object_points = np.array(common_object_points, dtype=np.float32).reshape(-1, 3)
                    common_left_corners = np.array(common_left_corners).reshape(-1, 1 ,2)
                    common_right_corners = np.array(common_right_corners).reshape(-1, 2)

                    # Pose estimation of the board to left camera
                    _, rvec_left, t_vect_left, _ = cv2.solvePnPRansac(
                        objectPoints=common_object_points,
                        imagePoints=common_left_corners,
                        cameraMatrix=self.K_1,
                        distCoeffs=self.d_1
                    )

                    # left to right camera
                    rvec_right, t_vect_right = cv2.composeRT(rvec_left, t_vect_left, cv2.Rodrigues(self.R)[0], self.T)[:2]

                    # project points to the right images
                    projected_points, _ = cv2.projectPoints(
                        objectPoints=common_object_points,
                        rvec=rvec_right,   
                        tvec=t_vect_right,                   
                        cameraMatrix=self.K_2,   
                        distCoeffs=self.d_2
                    )

                    projected_points = projected_points.reshape(-1, 2)

                    # visualization
                    errors_image = []
                    img_right = right_frame['image'].copy()
                    for actual, projected in zip(common_right_corners, projected_points):
                        cv2.circle(img_right, tuple(projected.astype(int)), 5, (0, 0, 255), -1) # projected point --> red
                        cv2.circle(img_right, tuple(actual.astype(int)), 5, (0, 255, 0), -1) # actual detected point --> green
                        cv2.line(img_right, tuple(map(int, actual)), tuple(map(int, projected)), (255, 0, 0), 1)
                        
                        # calculate error
                        error = np.mean(np.sqrt(np.sum((projected.reshape(-1, 2) - actual) ** 2, axis=1)))
                        projection_errors.append(error)
                        errors_image.append(error)
                    
                    all_projection_errors.append(np.mean(errors_image))
                    
                    cv2.imwrite(os.path.join(self.output_path, 'Projection_NonOverlap', right_frame['filename']), img_right)
        
        mean_reprojection_error = np.mean(projection_errors)

        # Visualization Reprojection Error per Image
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(all_projection_errors)), all_projection_errors, color='blue', alpha=0.7)
        plt.axhline(mean_reprojection_error, color='r', linestyle='dashed', linewidth=2, label="Mean projection Error")
        plt.title(f"Projection Error per Image ( Mean: {mean_reprojection_error:.4f} pixels )")
        plt.xlabel("Image index")
        plt.ylabel("Projection Error (pixels)")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_path, 'StereoAnalysis_NonOverlap/projection_error.png'))

        print("Visualizing projection done!")

        # Camera Pose Visualization
        camera_poses = []
    
        left_camera_pose = np.eye(4)
        camera_poses.append(left_camera_pose)

        right_camera_pose = np.eye(4)
        right_camera_pose[:3, :3] = self.R
        right_camera_pose[:3, 3] = self.T.T

        cam2_position_world = -np.dot(self.R.T, self.T)
        cam2_orientation_world = self.R.T

        right_camera_pose[:3, :3] = cam2_orientation_world
        right_camera_pose[:3, 3] = cam2_position_world.T

        camera_poses.append(right_camera_pose)

        # Calculate camera poses axis limits
        positions = np.array([pose[:3, 3] for pose in camera_poses])  # Extract positions from poses
    
        # Calculate min and max for each axis
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        
        # Add margin
        margin = 0.3
        xlim = (x_min - margin, x_max + margin)
        ylim = (y_min - margin, y_max + margin)
        zlim = (z_min - margin, z_max + margin)

        visualizer = CameraPoseVisualizer(xlim=xlim, ylim=ylim, zlim=zlim, output_path=self.output_path)

        labels = [f'Camera {i+1}' for i in range(len(camera_poses))]

        for idx, pose in enumerate(camera_poses):
            color = plt.cm.rainbow(idx / len(camera_poses))  # Assign a unique color to each camera
            visualizer.extrinsic2pyramid(pose, color=color, label=labels[idx])

        visualizer.show()

        print("Visualizing stereo camera poses done!\n")
    
    def save(self):
        """
        Save the non-overlap stereo calibration results
        """

        file_storage = cv2.FileStorage(os.path.join(self.output_path,'NonOverlapStereoCalibration.yaml'), cv2.FILE_STORAGE_WRITE)

        file_storage.write("K_1", self.K_1)
        file_storage.write("D_1", self.d_1)

        file_storage.write("K_2", self.K_2)
        file_storage.write("D_2", self.d_2)

        file_storage.write("R", self.R)
        file_storage.write("T", self.T)

        file_storage.release()

        print(f"Non-Overlap Stereo Calibration results saved in NonOverlapStereoCalibration.yaml")
    
    def calibNonOverlapStereo(self, save=False):
        """
        Non-overlap Stereo calibration
        """

        self.CheckCommonPoints()
        self.NonOverlapStereoCalibration()
        self.Rectification()
        self.visualization()
        
        if save:
            self.save()


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim, output_path):
        self.output_path = output_path
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=0.1, aspect_ratio=0.5, label=None):
        """
        Visualize a camera pose as a pyramid.
        """
        vertex_std = np.array([[0, 0, 0, 1],  # Camera origin
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2][:-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.6))
        
        if label:
            self.ax.text(vertex_transformed[0, 0], vertex_transformed[0, 1], vertex_transformed[0, 2], label, color=color, fontsize=10)
        
        # Draw axes for the camera
        origin = vertex_transformed[0, :-1]
        x_axis = origin + extrinsic[:3, 0] * focal_len_scaled
        y_axis = origin + extrinsic[:3, 1] * focal_len_scaled
        z_axis = origin + extrinsic[:3, 2] * focal_len_scaled

        self.ax.quiver(*origin, *(x_axis - origin), color='r', label='X-axis', length=1)
        self.ax.quiver(*origin, *(y_axis - origin), color='g', label='Y-axis', length=1)
        self.ax.quiver(*origin, *(z_axis - origin), color='b', label='Z-axis', length=1)

    def show(self):
        plt.title('Camera Extrinsics Visualization')
        plt.savefig(os.path.join(self.output_path, 'StereoAnalysis_NonOverlap/Stereo_camera_pose.png'))