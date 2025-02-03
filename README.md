# Grasp Transfer via RFP

## Setup

**Installation:**
Please use the provided `env.yml` to create a conda environment for this repository. The code is tested with the following:

- Python 3.10

- Pytorch 2.3

- Cuda 11.8

- Dependencies: `torch, numpy, scipy, tqdm, trimesh, plotly, pytorch_kinematics, transforms3d, numpy-quaternion`

We have the `mano_pybullet` added as a submodule which you can install by following steps. This module is included since we used this create a mano hand urdf from the original mano models. It also gives some utility functions to convert the mano hand parameters.

- `cd mano_pybullet`

- `pip install -e .`

- Please go through its README and test the functionality using the `gui_control` tool. You will need to set the `MANO_MODELS_DIR` env var to the path for extracted mano models dir.

  - If you see an error like `ImportError: cannot import name 'bool' from 'numpy'`:
    - Try: `pip install git+https://github.com/mattloper/chumpy` [Link to github issue](https://github.com/mattloper/chumpy/issues/55)


**Code Setup:**

- The core functionality is implemented in `model/`, specifically `hand_model.py` and `hand_opt.py`.

  - `hand_model.py` creates a differentiable kinematics model for a gripper given its URDF

    - Can use the [`GcsHandModel`](./model/hand_model.py/) defined in it as a standalone separately if needed!

  - `hand_opt.py` poses the grasp transfer as an optimization problem and provides wrappers for both logging and optimization.

    - The wrappers are for convinience, and the core optimization loop defined in [`GcsGraspTransferOpt`]('./model/hand_opt.py')

- `utils` includes some commonly used functions, importantly there are some utilities which can help with pose alignment between different grippers, and some rotation conversions.

  - `utils/grasp_utils.py`: gripper pose alignment to a common space -- useful for transferring grasps. Note, the values for each gripper are tuned according to the urdf models provided under `grippers/` dir. If your urdf is different from the ones provided, then you may need to define a custom alignment function: (1) hand palm normal should be +Z, (2) major axis for palm should be +Y, (3) hand origin should be on palm surface

- Gripper urdfs are under `grippers/`. Also included are files like:

  - `mgg_gripper_surface_pts.pk`: pickled dict containg the pre-selected interior surface points for the gripper along with their unified coordinates used for correspondence and transfer.

  - NOTE: The mano hand urdfs were created using the `mano_pybullet` repository.

  - And some other files for legacy reasons...

## Usage and Examples

- **Gripper Open/Close Heuristice:**
  - See `notebooks/example_find_grasp_frame.ipynb`
  - Modify it to store the open/close frames in a json file in the demonstration data dir. 
  - or any other representation as needed!
  - NOTE: if using an pre-existing conda env (`mfg2`) for this repo, you just need to update via pip:
    - `networkx==3.4.2`
    - `opencv-python`

- Primary Script: `transfer_from_hamer.py`. The main argument will be the demonstration data dir which has subfolders like `rgb, depth, pose` as well as `out/hamer/` for mano hand data from hamer 
  
  - `--input_dir`: path to demo data dir
  - `--mano_model_dir`: path to Mano `models/` dir, for example `~/Datasets/MANO/MANO_Hand_Model/mano_v1_2/models`
  - (Optional) `--target_gripper`: `fetch_gripper` by default
  - (Optional) `--debug_plots`: whether to save extra plotly html plots for vizualization (off by default) 

- Please see `notebooks/example_grasp_transfer.ipynb` for a usage example on grasp transfer. Playground for hamer transfer is under `notebooks/transfer_from_hamer.ipynb`.

  - The grasp transfer is supported between robot grippers under `grippers/` dir. 
  
  - The input to grasp transfer object `GcsGraspTransferOpt` requires: (1) source and target gripper names, (2) source gripper grasp q

  - Here grasp `q` refers to a `(9+d)` dimensional tensor where its broken down as:

    - `q[0:3]`: gripper base link translation vector with the grasp
    - `q[3:9]`: gripper base link orientation, represented as a 6d vector of two orthogonal components (think first 2 columns of a rotation matrix, in order like {x1,x2,x3,y1,y2,y3})
    - `q[9:d]`: joint values for `d` joints on the source gripper (so in essence `d ~ DOFS`)

  - It also includes some example to visualize the grippers and results. 

### To Do

- Add documentation for how we deal with left hand detections from hamer.

- Check if we need to tweak the correspondences between mano and fetch so that one finger is with thumb and other with middle finger? -- for better transfers and grasping? 

- Run `pip install --upgrade networkx` if urchin URDF loading gives an error.

### Other Ideas

Idea - 1:
- shoot rays from current gripper pose surface pts to object point cloud
- check for intersecting obj pts
- check for alignment between ray and pt normal (to eliminate colliding obj pts)
- utilize the filtered uv map on obj surface for the optimization

Idea - 2:
- check for obj pts in the inside region of gripper finger space
- optimize to get almost all obj pts inside the region
- only allow trans in y, z dirns
- only allow rotation around palm normal axis (+x for fetch gripper)

## References

The code and idea is adapted from the RobotFingerPrint paper which introduces a unified coordinate system over the gripper interior points.

```bibtex
@inproceedings{khargonkar2024robotfingerprint,
title={RobotFingerPrint: Unified Gripper Coordinate Space for Multi-Gripper Grasp Synthesis​},
author={Khargonkar, Ninad and Casas, Luis Felipe and  and Prabhakaran, Balakrishnan and Xiang, Yu},
journal={arXiv preprint arXiv:2409.14519},
year={2024}
}
```
