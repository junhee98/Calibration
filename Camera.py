import os
import cv2
import cv2.aruco as aruco
import glob
import numpy as np
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, root_path, output_path, image_prefix, image_format, aruco_dict, charuco_board, board_):
        # Iamge
        self.root_path = root_path
        self.image_shape = None
        self.image_prefix = image_prefix
        self.image_format = image_format
        self.image_shape = None

        # Board
        self.aruco_dict = aruco_dict
        self.charuco_board = charuco_board
        self.board = board_

        # Frames
        self.frames = []
        self.all_corners = []
        self.all_ids = []

        self.intrinsic_all_corners = []
        self.intrinsic_all_ids = []
        self.all_obj_points = []
        self.images = []
        self.filenames = []
        self.rvecs = None
        self.tvecs = None

        # Camera parameters
        self.K = None
        self.d = None

        # Output
        self.output_path = output_path
    
    def detectCharucoCornersAndIds(self, image, filename):
        """
        Detect charuco corners & ids
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        # Refine detected markers
        if ids is not None:
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(gray, self.charuco_board, corners, ids, None, None, None)

            # Draw detected markers
            img_marked = aruco.drawDetectedMarkers(image.copy(), corners, ids)

            # Detect charuco corners
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)

            if charuco_ids is not None:
                # Draw detected charuco corners
                img_marked = aruco.drawDetectedCornersCharuco(img_marked, charuco_corners, charuco_ids)

                # Save image with detected corners
                cv2.imwrite(os.path.join(self.output_path, self.image_prefix, 'Corners', filename), img_marked)

                # if len(charuco_corners) >= 6: # at least 6 corners are required for DLT algorithm in calibration
                if len(charuco_corners) >= (self.board.board_size[0] * self.board.board_size[1]) *0.3: # all corners are required for calibration
                    # Find object points for calcuate reproection error
                    objpoints = []
                    for charuco_id in charuco_ids.flatten():
                        objpoints.append(self.board.charuco_3d_points[charuco_id])

                    return "true", charuco_corners, charuco_ids, np.array(objpoints, dtype=np.float32)
                else:
                    return "few_corners", None, None, None
            else:
                return "non_corners", None, None, None

        return "non_markers", None, None, None

    def initFrame(self):
        """
        Detect charuco corners each images & save corners in dictionary
        """

        images = sorted(glob.glob(os.path.join(self.root_path, f'{self.image_prefix}', f'*.{self.image_format}')))

        print(f"=====Found {len(images)} images for {self.image_prefix}=====\n")

        for idx, image in enumerate(images):
            # Read image
            img = cv2.imread(image)

            # Extract filename for image-wise visualization
            filename = os.path.basename(image)

            if self.image_shape is None:
                self.image_shape = img.shape[:2][::-1]

            # Detect charuco corners & ids
            state, charuco_corners, charuco_ids, objpoints = self.detectCharucoCornersAndIds(img, filename)

            frame_info = {
                "state": state,
                "image": img,
                "idx": idx,
                "filename": filename,
                "charuco_corners": charuco_corners,
                "charuco_ids": charuco_ids,
            }
            self.frames.append(frame_info)
            self.all_corners.append(charuco_corners)
            self.all_ids.append(charuco_ids)

            if state == "true":
                print(f"Corners detected in {idx}th image in {self.image_prefix} : {len(charuco_corners)}")
                self.intrinsic_all_corners.append(charuco_corners)
                self.intrinsic_all_ids.append(charuco_ids)
                self.all_obj_points.append(objpoints)
                self.images.append(img.copy())
                self.filenames.append(filename)

            elif state == "few_corners":
                print(f"Few corners detected in {idx}th image in {self.image_prefix}")
                continue

            elif state == "non_corners":
                print(f"No corners detected in {idx}th image in {self.image_prefix}")
                continue

            elif state == "non_markers":
                print(f"No markers detected in {idx}th image in {self.image_prefix}")
                continue

            else:
                raise ValueError("Invalid state")
            
    def initCalibration(self):
        """
        Conduct Camera Calibration
        """

        # Initialize camera matrix
        init_K, init_d = self.initCameraIntrinsic(self.image_shape)

        # Calibrate camera
        ret, self.K, self.d, self.rvecs, self.tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=self.intrinsic_all_corners, 
            charucoIds=self.intrinsic_all_ids, 
            board=self.charuco_board, 
            imageSize=self.image_shape, 
            cameraMatrix=init_K,
            distCoeffs=init_d,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )

        if ret:
            print(f"\nCamera Intrinsic Calibration is done for {self.image_prefix}")
        else:
            raise ValueError(f"\nCamera calibration is failed for {self.image_prefix}")

    def initCameraIntrinsic(self, image_size):
        """
        Initialize camera intrinsic parameters
        """
        
        w, h = image_size
        fx = w
        fy = h
        cx = w // 2
        cy = h // 2

        init_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        init_d = np.zeros((5, 1), dtype=np.float32)
        
        return init_K, init_d

    def save(self):
        """
        Save camera parameters
        """
        
        file_storage = cv2.FileStorage(os.path.join(self.output_path, f'IntrinsicCalibration_{self.image_prefix}.yaml'), cv2.FILE_STORAGE_WRITE)

        file_storage.write("width", self.image_shape[0])
        file_storage.write("height", self.image_shape[1])

        file_storage.write("K", self.K)
        file_storage.write("D", self.d)

        file_storage.release()

        print(f"Camera Intrinsic Calibration results saved in IntrinsicCalibration_{self.image_prefix}.yaml\n")

    def visualize(self):
        """
        Visualize for Analyzing
        """
        # Make directory for visualization
        
        # Visualization for Reprojection in images
        if not os.path.exists(os.path.join(self.output_path, self.image_prefix, 'Reprojection')):
            os.makedirs(os.path.join(self.output_path, self.image_prefix, 'Reprojection'), exist_ok=True)

        # Visualization for Analyzing
        # Whole detected corners ( Whole images )
        # Reprojection error ( image-wise )
        # Reprojection error ( whole images )
        if not os.path.exists(os.path.join(self.output_path, self.image_prefix, 'Analysis')):
            os.makedirs(os.path.join(self.output_path, self.image_prefix, 'Analysis'), exist_ok=True)


        reprojection_errors = []
        x_errors, y_errors = [], []
        total_reprojection_error = 0
        total_points = 0

        all_corners_vis = np.vstack(self.intrinsic_all_corners).reshape(-1, 2)

        # Visualize detected points on virtual image plane
        plt.figure(figsize=(12, 6))
        plt.scatter(all_corners_vis[:, 0], all_corners_vis[:, 1], c='blue', label='Detected Points', s=10, alpha=0.7)
        plt.title(f'Detected {self.image_prefix} Corner Points on Virtual Image Plane')
        plt.xlabel('X-coordinate (pixels)')
        plt.ylabel('Y-coordinate (pixels)')
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.savefig(os.path.join(self.output_path, self.image_prefix, 'Analysis/Detected_points.png'))

        print(f"\nVisualizing detected corners in {self.image_prefix} is done")

        # Visualize Reprojection Error ( image-wise visualization )
        for i, (image, filename, rvec, tvec) in enumerate(zip(self.images, self.filenames, self.rvecs, self.tvecs)):
            # project 3D points to image plane
            projected_points, _ = cv2.projectPoints(
                objectPoints = self.all_obj_points[i],
                rvec = rvec,
                tvec = tvec,
                cameraMatrix = self.K, 
                distCoeffs = self.d
            )

            projected_points = projected_points.reshape(-1, 2)
            actual_points = self.intrinsic_all_corners[i].reshape(-1, 2)

            # edge case ( projected point are out of image plane )
            # valid_mask = (
            #     (projected_points[:, 0] >= 0) & (projected_points[:, 0] < self.image_shape[0]) &
            #     (projected_points[:, 1] >= 0) & (projected_points[:, 1] < self.image_shape[1])
            # )

            # projected_points = projected_points[valid_mask]
            # actual_points = actual_points[valid_mask]

            # Calculate reprojection error
            error = np.linalg.norm(projected_points - actual_points, axis=1) ** 2
            reprojection_errors.append(np.sqrt(np.sum(error) / len(error))) # Reprojection error for each image

            # Reprojection error for whole images
            total_reprojection_error += np.sum(error)
            total_points += len(actual_points)

            # Calculate x, y errors
            errors_points = projected_points - actual_points
            x_errors.extend(errors_points[:, 0])
            y_errors.extend(errors_points[:, 1])

            for actual, projected in zip(actual_points, projected_points):
                cv2.circle(image, tuple(map(int, actual)), 6, (0, 255, 0), -1) # actual point -> green
                cv2.circle(image, tuple(map(int, projected)), 5, (0, 0, 255), -1) # reprojected point -> red
                cv2.line(image, tuple(map(int, actual)), tuple(map(int, projected)), (255, 0, 0), 1) # btw actual and reprojected -> blue
            
            cv2.imwrite(os.path.join(self.output_path, self.image_prefix, 'Reprojection', filename), image)
        
        mean_error = np.sqrt(total_reprojection_error / total_points)

        # Visualize Reprojection Error ( each images with bar plot )
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(reprojection_errors)), reprojection_errors)
        plt.axhline(mean_error, color='r', linestyle='dashed', linewidth=2, label="RMS")
        plt.title(f"Reprojection Error per Image ( RMS: {mean_error:.4f} pixels )")
        plt.xlabel("Image index")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig(os.path.join(self.output_path, self.image_prefix, 'Analysis/Reprojection_errors_(image_wise).png'))

        # Visualize Reprojection Error ( whole images with Scatter Plot )
        plt.figure(figsize=(8, 6))
        plt.scatter(x_errors, y_errors, alpha=0.7, edgecolors='k', s=20, label="Reprojection Errors")
        plt.axhline(0, color='r', linestyle='dashed', linewidth=1, label="Ideal Line (Y=0)")
        plt.axvline(0, color='r', linestyle='dashed', linewidth=1)
        plt.title(f"Left Camera Reprojection Errors (pixels)")
        plt.xlabel("X Errors (pixels)")
        plt.ylabel("Y Errors (pixels)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_path, self.image_prefix, 'Analysis/Reprojection_errors_(scatter_plot).png'))

        print(f"Visualizing reprojection errors in {self.image_prefix} is done\n")

    def calibIntrinsic(self, save=False):
        """
        Camera Intrinsic Calibration
        """
        # Visualization detected corners in images
        if not os.path.exists(os.path.join(self.output_path, self.image_prefix, 'Corners')):
            os.makedirs(os.path.join(self.output_path, self.image_prefix, 'Corners'), exist_ok=True)

        self.initFrame() # Detect charuco corners in each images

        self.initCalibration() # Conduct Camera Calibration

        self.visualize() # Visualize for Analyzing
        
        if save: # Save camera parameters
            self.save()