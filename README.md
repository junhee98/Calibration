# Camera Calibration

<!-- TOC -->


This repo contains a implementation of Camera-Calibration. ( In this project, using Charuco board !)
* Monocular
* Stereo
* Non-overlap
***
## Install environment

```bash
conda env create --file calib.yaml
```


## Dataset structure
To conduct calibration, Put the set of images you want to calibrate into a folder.
In this repo, puts the set of images want to calibrate in `root_path/Cam_00x/*.png`.

* __Dataset structure ( Monocular )__
```bash
root_path
|
|--- Cam_001
  |
  |---0000.png
  |---0001.png
  |---0002.png
  |---...
|--- Cam_002
  |
  |---0000.png
  |---0001.png
  |---0002.png
  |---...
|--- Cam_00x
  ...
```

* __Dataset structure ( Stereo & Non-overlap )__
```bash
root_path
|
|--- Cam_001
  |
  |---0000.png
  |---0001.png
  |---0002.png
  |---...
|--- Cam_002
  |
  |---0000.png
  |---0001.png
  |---0002.png
  |---...
```
## Running

```bash
python main.py --calibration_version stereo --instrinsic False --img_prefix Cam --img_format png --root_path dataset/root/path --output_path output/path --aruco_dict DICT_6X6_250 --board_size "(6, 9)" --aruco_square 0.083 --aruco_marker 0.062
```

Here are the argument for running calibration code:
```
--calibration_version   Select calibration version ( mono | stereo | non_overlap_stereo )
--intrinsic             For stereo and non-overlap, if you have camera instrinsic parameter, using calibrated intrinsic parameter. (default = False )
--img_prefix            Folder prefix ( in this project, default = Cam )
--img_format            Image format ( in this project, default = png )
--root_path             Dataset root path ( check Dataset structure )
--output_path           Output directory ( default = result )
--aruco_dict            Dictionary setting of the aruco marker ( default = DICT_6X6_250 )
--board_size            Calibration board shape ( default = (6, 9) )
--aruco_square          Charuco board's squre size ( default = 0.083 )
--aruco_marker          Aruco marker's size ( default = 0.062 )
```
## More information

For using additional board setting, add board class in `Board.py`
```bash
class CharucoBoard_6_9_27_53: # like this!
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
```
Next, change board instance with updated board class
```bash
board = CharucoBoard_6_9_0_26(args.aruco_dict, args.aruco_square, args.aruco_marker) # like this!
```


## Todo

1. Make sequential calibration for low-overlapping setting.
2. Code cleaning.
