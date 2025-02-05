import os
import cv2
import glob
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class StereoCalib:
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

        self.R = None
        self.T = None
        self.E = None
        self.F = None

        self.map1_x = None
        self.map1_y = None
        self.map2_x = None
        self.map2_y = None

    def CheckCommonPoints(self):
        """
        Check common points between left and right cameras
        """
        # check total length of frames
        left_images = sorted(glob.glob(os.path.join(self.left_camera.root_path, f'{self.left_camera.image_prefix}', f'*.{self.left_camera.image_format}')))
        right_images = sorted(glob.glob(os.path.join(self.right_camera.root_path, f'{self.right_camera.image_prefix}', f'*.{self.right_camera.image_format}')))

        if len(left_images) != len(right_images):
            raise ValueError("The number of images in left and right cameras are different!")
        
        # Find same idx for left and right images
        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['idx'] != right_frame['idx']:
                raise ValueError("The indexes of left and right images are different!")
                
            if left_frame['state'] != right_frame['state']:
                print(f"The states of left and right images are different! -> left : {left_frame['state']}, right : {right_frame['state']}")
                continue
            
            if left_frame['state'] == 'true' and right_frame['state'] == 'true':
                # Find common points
                common_ids = np.intersect1d(left_frame['charuco_ids'].flatten(), right_frame['charuco_ids'].flatten())

                if len(common_ids) == 0:
                    print("No common corners detected between the two cameras for this frame. Skipping frame.")
                    continue

                if len(common_ids) >= 6: # at least 6 common points for stereo calibration
                    obj_points = self.left_camera.board.charuco_3d_points[common_ids]
                    img_pts_left = np.array([left_frame['charuco_corners'][left_frame['charuco_ids'].flatten() == id][0] for id in common_ids])
                    img_pts_right = np.array([right_frame['charuco_corners'][right_frame['charuco_ids'].flatten() == id][0] for id in common_ids])

                    self.common_3d_points.append(obj_points)
                    self.common_left_corner_points.append(img_pts_left)
                    self.common_right_corner_points.append(img_pts_right)

                else:
                    print("Few common corners detected between the two cameras for this frame. Skipping frame.")
                    continue

    def StereoCalibration(self):
        """
        Stereo calibration
        """
        # Stereo calibration
        ret, self.K_1, self.d_1, self.K_2, self.d_2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objectPoints = self.common_3d_points,
            imagePoints1 = self.common_left_corner_points,
            imagePoints2 = self.common_right_corner_points,
            cameraMatrix1 = self.K_1,
            distCoeffs1 = self.d_1,
            cameraMatrix2 = self.K_2,
            distCoeffs2 = self.d_2,
            imageSize = self.image_shape,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-6)
        )

        if ret:
            print("Stereo calibration done!\n")
        else:
            raise ValueError("Stereo calibration failed!\n")
        
    def Rectification(self):
        """
        Rectification
        """

        # Conduct stereo rectification
        R_1, R_2, P_1, P_2, Q, roi_1, roi_2 = cv2.stereoRectify(
            cameraMatrix1 = self.K_1,
            distCoeffs1 = self.d_1,
            cameraMatrix2 = self.K_2,
            distCoeffs2 = self.d_2,
            imageSize = self.image_shape,
            R = self.R,
            T = self.T,
            # flags=cv2.CALIB_ZERO_DISPARITY,
            flags=0,
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
        if not os.path.exists(os.path.join(self.output_path, self.left_camera.image_prefix, 'Rectified')):
            os.makedirs(os.path.join(self.output_path, self.left_camera.image_prefix, 'Rectified'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, self.right_camera.image_prefix, 'Rectified')):
            os.makedirs(os.path.join(self.output_path, self.right_camera.image_prefix, 'Rectified'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, 'Rectified')):
            os.makedirs(os.path.join(self.output_path, 'Rectified'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, 'Projection')):
            os.makedirs(os.path.join(self.output_path, 'Projection'), exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path, 'StereoAnalysis')):
            os.makedirs(os.path.join(self.output_path, 'StereoAnalysis'), exist_ok=True)

        rectified_y_errors = []
        all_y_errors = []

        reprojection_errors = []
        total_points = 0
        total_reprojection_error = 0

        aruco_dict_left = self.left_camera.board.aruco_dict
        aruco_dict_right = self.right_camera.board.aruco_dict

        board_left = self.left_camera.board.board
        board_right = self.right_camera.board.board

        # Visualization Rectified images
        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['state'] == 'true' and right_frame['state'] == 'true':
                # Rectify images
                left_rectified = cv2.remap(left_frame['image'].copy(), self.map1_x, self.map1_y, cv2.INTER_LINEAR)
                right_rectified = cv2.remap(right_frame['image'].copy(), self.map2_x, self.map2_y, cv2.INTER_LINEAR)

                # Save rectified images ( Each images )
                cv2.imwrite(os.path.join(self.output_path, self.left_camera.image_prefix, 'Rectified', left_frame['filename']), left_rectified)
                cv2.imwrite(os.path.join(self.output_path, self.right_camera.image_prefix, 'Rectified', right_frame['filename']), right_rectified)

                # Save rectified images ( Combined )
                combined = np.hstack((left_rectified, right_rectified))
                for line in range(0, combined.shape[0], 50): # draw horizontal lines with 50 pixel intervals
                    cv2.line(combined, (0, line), (combined.shape[1], line), (0, 255, 0), 1)
                
                cv2.imwrite(os.path.join(self.output_path, 'Rectified', left_frame['filename']), combined)

                # Calculate Rectification Error
                gray_left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
                gray_right_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

                corners_left_rectified, ids_left_rectified, _ = aruco.detectMarkers(gray_left_rectified, aruco_dict_left)
                corners_right_rectified, ids_right_rectified, _ = aruco.detectMarkers(gray_right_rectified, aruco_dict_right)

                corners_left_rectified, ids_left_rectified, _, _ = aruco.refineDetectedMarkers(
                    gray_left_rectified, 
                    board_left, 
                    corners_left_rectified, 
                    ids_left_rectified, 
                    None, 
                    None, 
                    None
                )

                corners_right_rectified, ids_right_rectified, _, _ = aruco.refineDetectedMarkers(
                    gray_right_rectified, 
                    board_right, 
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
                        board_left
                    )

                    _, charuco_corners_right_rectified, charuco_ids_right_rectified = aruco.interpolateCornersCharuco(
                        corners_right_rectified, 
                        ids_right_rectified, 
                        gray_right_rectified, 
                        board_right
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
        plt.savefig(os.path.join(self.output_path, 'StereoAnalysis/rectification_y_errors.png'))

        print("Visualizing rectified images and rectification errors done!")

        # Projection Common points in left image to right image
        for left_frame, right_frame in zip(self.left_camera.frames, self.right_camera.frames):
            if left_frame['state'] == 'true' and right_frame['state'] == 'true':
                # Find common points
                common_ids = np.intersect1d(left_frame['charuco_ids'].flatten(), right_frame['charuco_ids'].flatten())

                if len(common_ids) > 6:
                    common_object_points = []
                    common_left_corners = []
                    common_right_corners = []

                    for common_id in common_ids:
                        obj_points = self.left_camera.board.charuco_3d_points[common_id]
                        idx_left = np.where(left_frame['charuco_ids'].flatten() == common_id)[0][0]
                        idx_right = np.where(right_frame['charuco_ids'].flatten() == common_id)[0][0]

                        common_object_points.append(obj_points)
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

                    error = np.linalg.norm(projected_points - common_right_corners, axis=1) ** 2
                    reprojection_errors.append(np.sqrt(np.sum(error) / len(error))) # Reprojection error for each image

                    # Reprojection error for whole images
                    total_reprojection_error += np.sum(error)
                    total_points += len(common_right_corners)

                    # visualization
                    img_right = right_frame['image'].copy()
                    for actual, projected in zip(common_right_corners, projected_points):
                        cv2.circle(img_right, tuple(actual.astype(int)), 5, (0, 255, 0), -1) # actual detected point --> green
                        cv2.circle(img_right, tuple(projected.astype(int)), 5, (0, 0, 255), -1) # projected point --> red
                        cv2.line(img_right, tuple(map(int, actual)), tuple(map(int, projected)), (255, 0, 0), 1)
                    
                    cv2.imwrite(os.path.join(self.output_path, 'Projection', right_frame['filename']), img_right)
        
        mean_reprojection_error = np.sqrt(total_reprojection_error / total_points)

        # Visualization Reprojection Error per Image
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(reprojection_errors)), reprojection_errors, color='blue', alpha=0.7)
        plt.axhline(mean_reprojection_error, color='r', linestyle='dashed', linewidth=2, label="Mean projection Error")
        plt.title(f"Projection Error per Image ( Mean: {mean_reprojection_error:.4f} pixels )")
        plt.xlabel("Image index")
        plt.ylabel("Projection Error (pixels)")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_path, 'StereoAnalysis/projection_error.png'))

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
        Save the stereo calibration results
        """

        file_storage = cv2.FileStorage(os.path.join(self.output_path,'StereoCalibration.yaml'), cv2.FILE_STORAGE_WRITE)
        
        file_storage.write("width", self.image_shape[0])
        file_storage.write("height", self.image_shape[1])

        file_storage.write("K_1", self.K_1)
        file_storage.write("D_1", self.d_1)

        file_storage.write("K_2", self.K_2)
        file_storage.write("D_2", self.d_2)

        file_storage.write("R", self.R)
        file_storage.write("T", self.T)
        file_storage.write("E", self.E)
        file_storage.write("F", self.F)

        file_storage.release()

        print(f"Stereo Calibration results saved in StereoCalibration.yaml")
    
    def calibStereo(self, save=False):
        """
        Stereo calibration
        """

        self.CheckCommonPoints()
        self.StereoCalibration()
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
        plt.savefig(os.path.join(self.output_path, 'StereoAnalysis/Stereo_camera_pose.png'))