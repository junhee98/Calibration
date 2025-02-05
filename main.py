import os
import shutil
import argparse
from Camera import *
from Stereo import *
from NonOverlapStereo import *
from Board import *

def parse_tuple(value):
    try:
        return tuple(map(int, value.strip("()").split(",")))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: '{value}'")

def main(args):
    # Make output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Monocular calibration
    if args.calibration_version == 'mono':
        # Check the number of cameras
        camera_num = 0
        for entry in os.listdir(args.root_path):
            entry_path = os.path.join(args.root_path, entry)
            if os.path.isdir(entry_path) and entry.startswith(args.img_prefix):
                camera_num += 1

        # Calibration for each camera
        # initialize board
        board = CharucoBoard_6_9_0_26(args.aruco_dict, args.aruco_square, args.aruco_marker)

        # calibrate!
        for i in range(1, camera_num+1):
            
            camera = Camera(
                root_path = args.root_path,
                output_path = args.output_path,
                image_prefix = f'{args.img_prefix}_{i:03d}',
                image_format = args.img_format,
                aruco_dict = board.aruco_dict,
                charuco_board = board.board,
                board_ = board
            )

            camera.calibIntrinsic(save=True)
    
    elif args.calibration_version == 'stereo':
        # Intrinsic Calibration for each camera

        # initialize board
        # board = CharucoBoard_6_9_0_26(args.aruco_dict, args.aruco_square, args.aruco_marker)
        board  = CharucoBoard_5_5_144_156(args.aruco_dict, args.aruco_square, args.aruco_marker)

        # left camera intrinsic calibration
        left_camera = Camera(
            root_path = args.root_path,
            output_path = args.output_path,
            image_prefix = f'{args.img_prefix}_001',
            image_format = args.img_format,
            aruco_dict = board.aruco_dict,
            charuco_board = board.board,
            board_ = board
        )

        if args.intrinsics:
            left_camera.calibIntrinsic(save=False)
        else:
            left_camera.calibIntrinsic(save=True)

        # right camera intrinsic calibration
        right_camera = Camera(
            root_path = args.root_path,
            output_path = args.output_path,
            image_prefix = f'{args.img_prefix}_002',
            image_format = args.img_format,
            aruco_dict = board.aruco_dict,
            charuco_board = board.board,
            board_ = board
        )

        if args.intrinsics:
            right_camera.calibIntrinsic(save=False)
        else:
            right_camera.calibIntrinsic(save=True)

        if args.intrinsics:
            print("Default intrinsics are used for stereo calibration")

            # Load intrinsic parameters
            file_storage = cv2.FileStorage(os.path.join(args.output_path, 'IntrinsicCalibration_Cam_001.yaml'), cv2.FILE_STORAGE_READ)

            K = file_storage.getNode("K").mat()
            D = file_storage.getNode("D").mat()

            file_storage.release()

            left_camera.K = K
            left_camera.D = D

            file_storage = cv2.FileStorage(os.path.join(args.output_path, 'IntrinsicCalibration_Cam_002.yaml'), cv2.FILE_STORAGE_READ)

            K = file_storage.getNode("K").mat()
            D = file_storage.getNode("D").mat()

            file_storage.release()

            right_camera.K = K
            right_camera.D = D

            print("Intrinsic parameters are loaded")

        # Stereo Calibration
        stereo = StereoCalib(left_camera, right_camera)

        stereo.calibStereo(save=True)

    elif args.calibration_version == 'non_overlap_stereo':
        # Intrinsic Calibration for each camera

        # initialize board
        board_left = CharucoBoard_6_9_0_26(args.aruco_dict, args.aruco_square, args.aruco_marker)
        board_right = CharucoBoard_6_9_27_53(args.aruco_dict, args.aruco_square, args.aruco_marker)

        # left camera intrinsic calibration
        left_camera = Camera(
            root_path = args.root_path,
            output_path = args.output_path,
            image_prefix = f'{args.img_prefix}_001',
            image_format = args.img_format,
            aruco_dict = board_left.aruco_dict,
            charuco_board = board_left.board,
            board_ = board_left
        )

        if args.intrinsics:
            left_camera.calibIntrinsic(save=False)
        else:
            left_camera.calibIntrinsic(save=True)

        # right camera intrinsic calibration
        right_camera = Camera(
            root_path = args.root_path,
            output_path = args.output_path,
            image_prefix = f'{args.img_prefix}_002',
            image_format = args.img_format,
            aruco_dict = board_right.aruco_dict,
            charuco_board = board_right.board,
            board_ = board_right
        )

        if args.intrinsics:
            right_camera.calibIntrinsic(save=False)
        else:
            right_camera.calibIntrinsic(save=True)

        if args.intrinsics:
            print("Default intrinsics are used for non-overlap stereo calibration")

            # Load intrinsic parameters
            file_storage = cv2.FileStorage(os.path.join(args.output_path, 'IntrinsicCalibration_Cam_001.yaml'), cv2.FILE_STORAGE_READ)

            K = file_storage.getNode("K").mat()
            D = file_storage.getNode("D").mat()

            file_storage.release()

            left_camera.K = K
            left_camera.D = D

            file_storage = cv2.FileStorage(os.path.join(args.output_path, 'IntrinsicCalibration_Cam_002.yaml'), cv2.FILE_STORAGE_READ)

            K = file_storage.getNode("K").mat()
            D = file_storage.getNode("D").mat()

            file_storage.release()

            right_camera.K = K
            right_camera.D = D

            print("Intrinsic parameters are loaded")

        # Stereo Calibration ( Non-overlapping ) --- small overlap setting is needed for rectification and proejction!!
        nonoverlapstereo = NonOverlapStereoCalib(left_camera, right_camera)

        nonoverlapstereo.calibNonOverlapStereo(save=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Calibration version
    parser.add_argument('--calibration_version', type=str, default='stereo')
    parser.add_argument('--intrinsics', type=bool, default=False)

    # Input & Output path
    parser.add_argument('--root_path', type=str, default='/root/dev/Calib/Calibration_study_code/datasets/MC_Calib_stereo_sampled')
    parser.add_argument('--output_path', type=str, default='MC_Calib_stereo_sampled_results')

    # Image prefix & format
    parser.add_argument('--img_prefix', type=str, default='Cam')
    parser.add_argument('--img_format', type=str, default='png')

    # Calibration parameters
    parser.add_argument('--aruco_dict', type=str, default='DICT_6X6_250')
    # parser.add_argument("--board_size", type=parse_tuple, default=(6, 9))
    # parser.add_argument('--aruco_square', type=float, default=0.083)
    # parser.add_argument('--aruco_marker', type=float, default=0.062)
    parser.add_argument("--board_size", type=parse_tuple, default=(5, 5))
    parser.add_argument('--aruco_square', type=float, default=0.04)
    parser.add_argument('--aruco_marker', type=float, default=0.03)
    
    args = parser.parse_args()

    main(args)

