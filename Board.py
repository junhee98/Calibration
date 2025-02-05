import numpy as np
import cv2.aruco as aruco

class CharucoBoard_6_9_0_26:
    def __init__(self, aruco_dict, square_length, marker_length):
        self.board_size = (6, 9)
        self.board_marker_ids = np.array(range(0, 27))

        self.aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict))
        
        self.board = aruco.CharucoBoard(
            (self.board_size[0], self.board_size[1]),  
            square_length,                 
            marker_length,                   
            self.aruco_dict,
            self.board_marker_ids                    
        )
        
        self.charuco_3d_points = self.board.getChessboardCorners()

class CharucoBoard_5_5_144_156:
    def __init__(self, aruco_dict, square_length, marker_length):
        self.board_size = (5, 5)
        self.board_marker_ids = np.array(range(144, 156))

        self.aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict))
        
        self.board = aruco.CharucoBoard(
            (self.board_size[0], self.board_size[1]),  
            square_length,                 
            marker_length,                   
            self.aruco_dict,
            self.board_marker_ids                    
        )
        
        self.charuco_3d_points = self.board.getChessboardCorners()


class CharucoBoard_6_9_27_53:
    def __init__(self, aruco_dict, square_length, marker_length):
        self.board_size = (6, 9)
        self.board_marker_ids = np.array(range(27, 54))

        self.aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict))
        
        self.board = aruco.CharucoBoard(
            (self.board_size[0], self.board_size[1]),  
            square_length,                 
            marker_length,                   
            self.aruco_dict,
            self.board_marker_ids                    
        )
        
        self.charuco_3d_points = self.board.getChessboardCorners()
