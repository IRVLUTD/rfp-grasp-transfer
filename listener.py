#!/usr/bin/env python
"""ROS image listener"""

import threading
import numpy as np

# from scipy.io import savemat

import rospy
import tf
import tf2_ros
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge

# from ros_utils import ros_qt_to_rt

from nav_msgs.msg import Odometry, OccupancyGrid
import time
import yaml
import ros_numpy

lock = threading.Lock()


def ros_qt_to_rt(quat, posn):
    """
    Converts ROS quaternion and position to a 4x4 transformation matrix.
    :param quat: List of quaternion [x, y, z, w]
    :param posn: List of position [x, y, z]
    :return: 4x4 numpy array
    """
    mat = tf.transformations.quaternion_matrix(quat)
    mat[:3, 3] = np.array(posn)
    return mat


class ImageListener:

    def __init__(self, camera="Fetch"):

        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        self.base_frame = "base_link"
        self.camera_frame = "head_camera_rgb_optical_frame"
        self.target_frame = self.base_frame

        self.tf_listener = tf.TransformListener()

        rgb_sub = message_filters.Subscriber(
            "/head_camera/rgb/image_raw", Image, queue_size=10
        )
        depth_sub = message_filters.Subscriber(
            "/head_camera/depth_registered/image_raw", Image, queue_size=10
        )

        pc_sub = message_filters.Subscriber(
            "/head_camera/depth_downsample/points", PointCloud2, queue_size=10
        )

        msg = rospy.wait_for_message("/head_camera/rgb/camera_info", CameraInfo)

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, pc_sub], queue_size, slop_seconds
        )
        ts.registerCallback(self.callback_rgbd)

    def callback_rgbd(self, rgb, depth, pc):

        # get camera pose in base
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )
            RT_camera = ros_qt_to_rt(rot, trans)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn("Update failed... " + str(e))
            RT_camera = None

        if depth.encoding == "32FC1":
            # depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0
        elif depth.encoding == "16UC1":
            # depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            # depth_cv = depth_cv.copy().astype(np.float32)
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1,
                "Unsupported depth type. Expected 16UC1 or 32FC1, got {}".format(
                    depth.encoding
                ),
            )
            return

        # im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        im = ros_numpy.numpify(rgb)
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
        # import pdb

        # pdb.set_trace()
        # Extract just xyz coordinates if needed
        xyz_array = ros_numpy.point_cloud2.get_xyz_points(pc_array)
        default_alpha = np.ones(
            (xyz_array.shape[0], 1), dtype=np.float32
        )  # Match float32 dtype
        xyz_array = np.column_stack((xyz_array, default_alpha))

        # xyz_array = pc_array  # if 4 channels
        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera
            self.pc_in_cam = xyz_array

    def get_data_to_save(self):

        with lock:
            if self.im is None:
                return None, None
            RT_camera = self.RT_camera.copy()
            pc_in_cam = self.pc_in_cam.copy()
        return (RT_camera, pc_in_cam)


if __name__ == "__main__":
    # test_basic_img()
    rospy.init_node("image_listener")
    listener = ImageListener()
    time.sleep(3)
