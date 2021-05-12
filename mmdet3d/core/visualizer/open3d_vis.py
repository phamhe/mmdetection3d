import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math

import os
import base64

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')


def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyz'):
    """Draw points on visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.

    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    return pcd, points_colors


def _draw_bboxes(bbox3d,
                 vis,
                 points_colors,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
    """Draw bbox on visualizer and change the color of points inside bbox3d.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`): point cloud. Default: None.
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points inside bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


def show_pts_boxes(points,
                   bbox3d=None,
                   show=True,
                   save_path=None,
                   points_size=2,
                   point_color=(0.5, 0.5, 0.5),
                   bbox_color=(0, 1, 0),
                   points_in_box_color=(1, 0, 0),
                   rot_axis=2,
                   center_mode='lidar_bottom',
                   mode='xyz'):
    """Draw bbox and points on visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize. Default: None.
        show (bool): whether to show the visualization results. Default: True.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes(bbox3d, vis, points_colors, pcd, bbox_color,
                     points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()

def get_corners(center, size, yaw):
    rot = np.asmatrix([[math.cos(yaw), -math.sin(yaw)],\
                    [math.sin(yaw),  math.cos(yaw)]])
    plain_pts = np.asmatrix([[0.5 * size[1], 0.5*size[0]],\
                        [0.5 * size[1], -0.5*size[0]],\
                        [-0.5 * size[1], -0.5*size[0]],\
                        [-0.5 * size[1], 0.5*size[0]]])
    tran_pts = np.asarray(rot * plain_pts.transpose());
    tran_pts = tran_pts.transpose()
    corners = np.arange(24).astype(np.float32).reshape(8, 3)
    for i in range(8):
        corners[i][0] = center[0] + tran_pts[i%4][1]
        corners[i][1] = center[1] + tran_pts[i%4][0]
        corners[i][2] = center[2] + (float(i >= 4) - 0.5) * size[2];
    return corners


def _draw_bboxes_ind(bbox3d,
                     vis,
                     indices,
                     points_colors,
                     pcd=None,
                     bbox_color=(0, 1, 0),
                     points_in_box_color=(1, 0, 0),
                     rot_axis=2,
                     center_mode='lidar_bottom',
                     mode='xyz'):
    """Draw bbox on visualizer and change the color or points inside bbox3d
    with indices.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        indices (numpy.array | torch.tensor, shape=[N, M]):
            indicate which bbox3d that each point lies in.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`): point cloud. Default: None.
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        # TODO: fix problem of current coordinate system
        # dim[0], dim[1] = dim[1], dim[0]  # for current coordinate
        # yaw[rot_axis] = -(bbox3d[i, 6] - 0.5 * np.pi)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)

        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            points_colors[indices[:, i].astype(np.bool)] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


def show_pts_index_boxes(points,
                         bbox3d=None,
                         show=True,
                         indices=None,
                         save_path=None,
                         points_size=2,
                         point_color=(0.5, 0.5, 0.5),
                         bbox_color=(0, 1, 0),
                         points_in_box_color=(1, 0, 0),
                         rot_axis=2,
                         center_mode='lidar_bottom',
                         mode='xyz'):
    """Draw bbox and points on visualizer with indices that indicate which
    bbox3d that each point lies in.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize. Default: None.
        show (bool): whether to show the visualization results. Default: True.
        indices (numpy.array | torch.tensor, shape=[N, M]):
            indicate which bbox3d that each point lies in. Default: None.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes_ind(bbox3d, vis, indices, points_colors, pcd, bbox_color,
                         points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


def project_pts_on_img(points,
                       raw_img,
                       lidar2img_rt,
                       max_distance=70,
                       thickness=-1):
    """Project the 3D points cloud on 2D image.

    Args:
        points (numpy.array): 3D points cloud (x, y, z) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        max_distance (float): the max distance of the points cloud.
            Default: 70.
        thickness (int, optional): The thickness of 2D points. Default: -1.
    """
    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=1,
            color=tuple(color),
            thickness=thickness,
        )
    cv2.imshow('project_pts_img', img)
    cv2.waitKey(100)


def project_bbox3d_on_img(bboxes3d,
                          raw_img,
                          lidar2img_rt,
                          color=(0, 255, 0),
                          thickness=1):
    """Project the 3D bbox on 2D image.

    Args:
        bboxes3d (numpy.array, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        color (tuple[int]): the color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_bbox):
        corners = imgfov_pts_2d[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    cv2.imshow('project_bbox3d_img', img)
    cv2.waitKey(0)


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7]): 3d bbox (x, y, z, dx, dy, dz, yaw)
            to visualize. The 3d bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path=None,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.

        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3D bbox (x, y, z, dx, dy, dz, yaw) to be visualized.
                The 3d bbox is in mode of Box3DMode.DEPTH with
                gravity_center (please refer to core.structures.box_3d_mode).
            bbox_color (tuple[float]): the color of bbox. Defaule: None.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d. Defaule: None.
        """
        if bbox_color is None:
            bbox_color = self.bbox_color
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def show(self, save_path=None):
        """Visualize the points cloud.

        Args:
            save_path (str): path to save image. Default: None.
        """

        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return

class Visualizer_bev(object):
    def __init__(self, point, 
                points_range,
                out_dir,
                prefix,
                save=True,
                scale_factor=[20, 20],
                padding=[4, 2],
                grid_factor=2,
                fix_width=1600):
        super(Visualizer_bev, self).__init__()
        self.out_dir = out_dir
        self.prefix = prefix
        self.save = save

        self.frame_size = np.zeros((2,), dtype=np.int32)
        self.frame_size[0] = int(points_range[4] - points_range[1]) + padding[1]
        self.frame_size[1] = int(points_range[3] - points_range[0]) + padding[0]
        self.scale_factor = scale_factor

        self.scale_factor[0] = int(fix_width/self.frame_size[1])
        self.scale_factor[1] = self.scale_factor[0]

        self.frame_size[1] *= self.scale_factor[0]
        self.frame_size[0] *= self.scale_factor[1]

        # prepare canvas
        self.canvas = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype='uint8')
        self.canvas.fill(255)
        canvas_center_x = int(self.frame_size[1]/2)
        canvas_center_y = int(self.frame_size[0]/2)

        cv2.circle(self.canvas,
                     (canvas_center_x,
                     canvas_center_y),
                     4, (0, 0, 0), -1)
        x_axis = int(canvas_center_x/(grid_factor*self.scale_factor[1]))
        y_axis = int(canvas_center_y/(grid_factor*self.scale_factor[0]))
        for i in range(-x_axis, x_axis+1):
            cv2.line(self.canvas, 
                        (canvas_center_x+i*grid_factor*self.scale_factor[0], 
                        0), 
                        (canvas_center_x+i*grid_factor*self.scale_factor[0], self.frame_size[0]),
                        (0, 255, 255), 1)
        for i in range(-y_axis, y_axis+1):
            cv2.line(self.canvas, 
                        (0, 
                        canvas_center_y+i*grid_factor*self.scale_factor[0]), 
                        (self.frame_size[1], 
                        canvas_center_y+i*grid_factor*self.scale_factor[1]),
                        (0, 255, 255), 1)
        class_name = ['PEDESTRIAN',
                        'CYCLIST',
                        'CAR',
                        'TRUCK',
                        'BUS']
        # key area 0
        # cv2.rectange(self.canvas, (),
        #                           (),
        #                           (0, 0, 255))

        self.frame_infos = {}
        for cls in class_name:
            self.frame_infos[cls] = {
                                    'gt_annos':[],
                                    'dts':[],
                                    'fps':[],
                                    'fns':[],
                                    'others':[]
                                    }

    def add_bboxes(self, bboxes_ori, color, labels, bbox_type):

        bboxes = bboxes_ori.copy()
        bboxes[:, 0] *= self.scale_factor[0]
        bboxes[:, 4] *= self.scale_factor[0]
        bboxes[:, 1] *= self.scale_factor[1]
        bboxes[:, 3] *= self.scale_factor[1]

        bboxes[:, 1] = -bboxes[:, 1]
        bboxes[:, 0] = bboxes[:, 0] + 20 + (self.frame_size[1])/2
        bboxes[:, 1] = bboxes[:, 1] + 20 + (self.frame_size[0])/2

        self.draw_bev_bboxes(bboxes, labels, color, bbox_type)

    def draw_bev_bboxes(self, bboxes, labels=None, color=(255, 0, 0), bbox_type=None, extra_infos=True, rot_axis=2):
        new_color = (color[2], color[1], color[0])
        for i, bbox in enumerate(bboxes):
            center = bboxes[i, 0:2]
            size = bboxes[i, 3:5]
            yaw = bboxes[i, 6]

            rot = np.asmatrix([[math.cos(yaw), -math.sin(yaw)],\
                                [math.sin(yaw),  math.cos(yaw)]])
            plain_pts = np.asmatrix([[0.5 * size[0], 0.5*size[1]],\
                                   [0.5 * size[0], -0.5*size[1]],\
                                   [-0.5 * size[0], -0.5*size[1]],\
                                   [-0.5 * size[0], 0.5*size[1]]])
            tran_pts = np.asarray(rot * plain_pts.transpose())
            tran_pts = tran_pts.transpose()
            corners = np.arange(8).astype(np.float32).reshape(4, 2)
            for j in range(4):
                corners[j][0] = center[0] + tran_pts[j][0]
                corners[j][1] = center[1] + tran_pts[j][1]
            corners = corners.reshape((-1, 1, 2))
            corners = corners.astype(dtype=np.int32)
            cv2.polylines(self.canvas, [corners], True, new_color, 2)

            if extra_infos:
                if labels is not None:
                    label = labels[i]
                    if bbox_type == 'gt_annos' or \
                        bbox_type == 'fns':
                        x_pos = int(bboxes[i, 0])
                        y_pos = int(bboxes[i, 1])
                        cv2.putText(self.canvas, 
                                '%s'%(i), 
                               (x_pos, y_pos), 
                               cv2.FONT_HERSHEY_COMPLEX, 0.4, new_color, 1)
                        label = np.concatenate((label, np.array([i])))
                    self.frame_infos[label[0]][bbox_type].append(label)

    def context_infos(self):
        fout = open(os.path.join(self.out_dir, 'frame_%s.txt'%self.prefix), 'w')
        out_str = 'RSULTS IN DETECTION\n'
        out_str += '----------------------------\n'
        for cls in self.frame_infos:
            out_str += '  %s\n'%cls
            cls_infos = self.frame_infos[cls]
            for b_type in cls_infos:
                out_str += '    %s\n'%b_type
                if len(cls_infos[b_type]) == 0:
                    continue
                labels = np.stack(cls_infos[b_type], 0)
                # inds = np.argsort(labels[:, 5])
                for idx in range(labels.shape[0]):
                    out_str += '      %s %.2f %.2f %.3f %.3f'%(
                    labels[idx, 0], 
                    float(labels[idx, 5]), 
                    float(labels[idx, 6]),
                    float(labels[idx, 2]),
                    float(labels[idx, 3]))
                    if labels[idx].shape[0] > 7:
                        out_str += ' %s\n'%labels[idx, 7]
                    else:
                        out_str += '\n'
            out_str += '-----------------------------\n'
        fout.write(out_str)
        fout.close()
        cv2.imwrite(os.path.join(self.out_dir, 'frame_%s.jpg'%self.prefix), self.canvas)
        
        

    def show(self):
        cv2.imshow('debug', self.canvas)
        cv2.waitKey()

    def decode_img(self):
        canvas_ = cv2.imencode('.jpg', self.canvas)[1]
        canvas_ = base64.b64encode(canvas_)
        canvas_ = canvas_.decode()
        canvas_ = "data:image/png;base64," + canvas_
        return canvas_
        


class Visualizer_html(object):
    def __init__(self, points,
                    key_area,
                    cam_cfg=None):
        super(Visualizer_html, self).__init__()
        self.points = points
        self.points_cam = None
        if cam_cfg is not None:
            self.cam_cfg = cam_cfg
        else:
            self.cam_cfg = {
                        'width':1920,
                        'height':1080,
                        'x_rad':-0.8, # + is shunshizhen
                        'y_rad':-np.pi/2,
                        'z_rad':0,
                        'trans':np.array([0, 0, 40])
                    }
        self.lidar_2_cam = np.array(
                                [[0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]]
                            ).astype(np.float32)
        cam_r_x  = np.array(
                           [[1, 0, 0],
                           [0, 
                               np.cos(self.cam_cfg['x_rad']), 
                               np.sin(self.cam_cfg['x_rad'])],
                           [0, 
                               -np.sin(self.cam_cfg['x_rad']), 
                               np.cos(self.cam_cfg['x_rad'])]],
                           )
        cam_r_y  = np.array(
                           [[np.cos(self.cam_cfg['y_rad']), 0,
                               -np.sin(self.cam_cfg['y_rad'])],
                           [0, 1, 0],
                           [np.sin(self.cam_cfg['y_rad']), 0,
                               np.cos(self.cam_cfg['y_rad'])]]
                           )
        cam_r_z  = np.array(
                           [[np.cos(self.cam_cfg['z_rad']),
                               -np.sin(self.cam_cfg['z_rad']), 0],
                             [np.sin(self.cam_cfg['z_rad']),
                               np.cos(self.cam_cfg['z_rad']), 0],
                             [0, 0, 1]]
                           )
        mat_r = cam_r_z @ cam_r_x @ cam_r_y
        self.cam_ex = np.concatenate((mat_r, 
                                        self.cam_cfg['trans'].reshape(3, 1)), 1)
        self.cam_ex = np.concatenate((self.cam_ex, 
                                        np.array([0, 0, 0, 1])\
                                            .reshape(1, 4)), 0)

        self.cam_in = np.array(
                            [[500, 0, 960, 0],
                            [0, 500, 540, 0],
                            [0, 0, 1, 0]]
                        ).astype(np.float32)
        self.mat_proj = self.cam_in @ self.cam_ex @ \
                            self.lidar_2_cam

        self.canvas = np.zeros((self.cam_cfg['height'], 
                                    self.cam_cfg['width'], 3), 
                                    dtype='uint8')
        self.canvas.fill(255)
        # draw key area
        self.key_area_mask(key_area)

    def key_area_mask(self, key_area):
        color_map = [
                        [
                            (255, 0, 255),
                            (238, 130, 238)
                        ],
                        [
                            (240, 32, 160),
                            (219, 112, 147)
                        ],
                    ]
        key_area_points = np.zeros((2, 2, 4, 4))
        key_area_points[:, :, :, 3] = 1
        for i in range(2):
            for j in range(2):
                key_area_points[i, j, 0, :2] = \
                    key_area[2*i, j, :2]
                key_area_points[i, j, 1, :2] = \
                    key_area[2*i, j, ::3]
                key_area_points[i, j, 2, :2] = \
                    key_area[2*i, j, 2:]
                key_area_points[i, j, 3, 0] = \
                    key_area[2*i, j, 2]
                key_area_points[i, j, 3, 1] = \
                    key_area[2*i, j, 1]
        key_area_cam = self.mat_proj @ key_area_points.reshape((-1, 4, 1))
        key_area_cam = key_area_cam.reshape((key_area_points.shape[0],
                                        key_area_points.shape[1],
                                        key_area_points.shape[2], -1))
        factor = key_area_cam[:, :, :, 2]\
                    .reshape(
                        (key_area_cam.shape[0],
                        key_area_cam.shape[1],
                        key_area_cam.shape[2], 1))
        factor = np.concatenate((factor, factor), 3)
        key_area_cam[:, :, :, :2] /= factor
        key_area_cam = key_area_cam[:, :, :, :2]
        mask = np.zeros_like(self.canvas)
        mask.fill(255)
        for i in range(key_area_cam.shape[0]-1,
                        -1, -1):
            for j in range(key_area_cam.shape[1]-1, 
                            -1, -1):
                corners = key_area_cam[i, j]
                corners = corners.reshape((-1, 1, 2))
                corners = corners.astype(np.int)
                cv2.polylines(mask, [corners],
                                True, color_map[i][j], 1)
                cv2.fillPoly(mask, [corners],
                                color_map[i][j])
        alpha = 0.7
        beta = 0.3
        gamma = 0
        self.canvas = cv2.addWeighted(self.canvas, 
                           alpha, mask, 
                           beta, gamma)

    def add_points(self):
         pts_num = self.points.shape[0]
         padding = np.ones((pts_num, 1), 
                            dtype=self.points.dtype)
         self.points = self.points[:, :3]
         self.points = np.concatenate((self.points, 
                                        padding), 1)


         points_ = self.points.reshape((-1, 4, 1))
         self.points_cam = self.mat_proj @ points_
         self.points_cam = self.points_cam.reshape((-1, 3))
         # temporoal debug code
         inds = self.points_cam[:, 2] > 0
         self.points_cam = self.points_cam[inds]

         factor = self.points_cam[:, 2].reshape((-1, 1))
         factor = np.concatenate((factor, factor), 1)
         self.points_cam[:, :2] /= factor
         self.points_cam = self.points_cam[:, :2]
         
         for point in self.points_cam:
            cv2.circle(self.canvas, (int(point[0]),
                                        int(point[1])),
                                        1, (220, 220, 220), -1)
    def add_bboxes(self, bboxes, color):
        new_color = (color[2], color[1], color[0])
        for bbox in bboxes:
            center = bbox[0:3]
            size = bbox[3:6]
            yaw = bbox[6]
            corners = get_corners(center, size, yaw)

            corners = np.concatenate((corners, 
                                        np.ones((
                                            corners.shape[0], 
                                            1))), 1)
            corners = self.mat_proj @ corners.reshape((-1, 4, 1))[:]
            corners = corners.reshape((-1, 3))
            factor = corners[:, 2].reshape((-1, 1))
            factor = np.concatenate((factor, factor), 1)
            corners[:, :2] /= factor
            corners = corners[:, :2]
            corners[:, 0] = np.clip(corners[:, 0], 0, self.canvas.shape[1])
            corners[:, 1] = np.clip(corners[:, 1], 0, self.canvas.shape[0])
            cv2.polylines(self.canvas, [corners[0:4].astype(np.int)], 
                                            True, new_color, 1)
            cv2.polylines(self.canvas, [corners[4:].astype(np.int)], 
                                            True, new_color, 1)
            for i in range(4):
                cv2.line(self.canvas, (int(corners[i][0]),
                                        int(corners[i][1])),
                                        (int(corners[i+4][0]),
                                        int(corners[i+4][1])),
                                        new_color, 1)

    def show(self):
        cv2.imshow('debug', self.canvas)
        cv2.waitKey()

    def decode_img(self):
        canvas_ = cv2.imencode('.jpg', self.canvas)[1]
        canvas_ = base64.b64encode(canvas_)
        canvas_ = canvas_.decode()
        canvas_ = "data:image/png;base64," + canvas_
        return canvas_
