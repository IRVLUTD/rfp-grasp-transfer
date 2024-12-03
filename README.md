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

- Please go through its README and test the functionality using the `gui_control` tool.

  - If you see an error like `ImportError: cannot import name 'bool' from 'numpy'`, please try: `pip install git+https://github.com/mattloper/chumpy` [Link to github issue](https://github.com/mattloper/chumpy/issues/55)


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

- Please see `notebooks/example_grasp_transfer.ipynb` for a usage example on grasp transfer. 

- The grasp transfer is supported between robot grippers under `grippers/` dir. 
  
- The input to grasp transfer object `GcsGraspTransferOpt` requires: (1) source and target gripper names, (2) source gripper grasp q

  - Here grasp `q` refers to a `(9+d)` dimensional tensor where its broken down as:

    - `q[0:3]`: gripper base link translation vector with the grasp
    - `q[3:9]`: gripper base link orientation, represented as a 6d vector of two orthogonal components (think first 2 columns of a rotation matrix, in order like {x1,x2,x3,y1,y2,y3})
    - `q[9:d]`: joint values for `d` joints on the source gripper (so in essence `d ~ DOFS`)

- It also includes some example to visualize the grippers and results. 

### To Do

- Add instructions and usage example to obtain the Mano Hand URDF's dofs and base link pose given the $(\theta, \beta)$ parameters.

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
