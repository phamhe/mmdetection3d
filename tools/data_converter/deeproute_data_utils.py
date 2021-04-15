import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io
import json
import math
import cv2

def get_image_index_str(img_idx, use_prefix_id=False):
    return '{:05d}'.format(img_idx)


def get_deeproute_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2',
                   use_prefix_id=False):
    return get_deeproute_info_path(idx, prefix, info_type , '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_deeproute_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check, use_prefix_id)



def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'hard_type':[]
    })
    with open(label_path, 'r') as f:
        objs = json.load(f)
    objs = objs['objects']
    name = []
    truncated = []
    occluded = []
    dimensions = []
    location = []
    rotation_y = []
    hard_type = []
    num_objects = 0
    for obj in objs:
        if obj['type'] != 'DontCare':
            num_objects += 1
        name.append(obj['type'])
        trun = 0
        if 'truncated' in obj:
            trun = obj['truncated']
        truncated.append(trun)
        occ = 0
        if 'occluded' in obj:
            occ = obj['occluded']
        occluded.append(occ)
        ht = 'easy'
        if 'hard_type' in obj:
            ht = obj['hard_type']
        hard_type.append(ht)

        dimensions.append([obj['bounding_box']['width'],
                            obj['bounding_box']['length'],
                            obj['bounding_box']['height']])
        location.append([obj['position']['x'],
                            obj['position']['y'],
                            obj['position']['z']])
        rotation_y.append(obj['heading'])

        
    annotations['name'] = np.array(name)
    annotations['truncated'] = np.array(truncated)
    annotations['occluded'] = np.array(occluded)
    annotations['hard_type'] = np.array(hard_type)
    annotations['dimensions'] = np.array(dimensions).reshape(-1, 3)
    annotations['location'] = np.array(location).reshape(-1, 3)
    annotations['rotation_y'] = np.array(rotation_y).reshape(-1)
    annotations['score'] = np.zeros((annotations['name'].shape[0], ))
    num_gt = annotations['name'].shape[0]

    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_deeproute_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         image_ids=7000,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    Deeproute annotation format version 2:
    {
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)

def get_corners(loc, dims, rot):
    center = loc[0:2]
    size = dims[0:2]
    if rot < 0:
        rot += np.pi
    yaw = rot
   
    rot = np.asmatrix([[math.sin(yaw), -math.cos(yaw)],\
                        [math.cos(yaw),  math.sin(yaw)]])
    plain_pts = np.asmatrix([[0.5 * size[1], 0.5*size[0]],\
                           [0.5 * size[1], -0.5*size[0]],\
                           [-0.5 * size[1], -0.5*size[0]],\
                           [-0.5 * size[1], 0.5*size[0]]])
    tran_pts = np.asarray(rot * plain_pts.transpose())
    tran_pts = tran_pts.transpose()
    corners = np.arange(8).astype(np.float32).reshape(4, 2)
    for j in range(4):
        corners[j][0] = center[0] + tran_pts[j][0]
        corners[j][1] = center[1] + tran_pts[j][1]
    # corners = corners.astype(dtype=np.int32)

    return corners

def angle_cos(a, b):
    return a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))  

def get_max_ang_corners(corners, 
                        loc, shape):
    O = shape/2
    corners_o = corners[:] - O
    loc_o = loc - O
    angles = []
    for corner in corners_o:
        angle = angle_cos(corner, loc_o)
        angles.append(angle)
    idxs = np.argsort(angles)
    idxs_res = idxs[:2]

    # if max
    res = corners_o[idxs_res]
    if (res[0, 0] - loc_o[0])*(res[1, 0] - loc_o[0]) > 0:
        idxs_res[1] = idxs[2]

    return corners[idxs_res]

def get_k(corners, shape):
    ks = []
    for corner in corners:
        a = np.ones((2, 2))
        b = np.zeros((2, 1))
        a[0, 0] = shape[0]/2
        b[0, 0] = shape[1]/2
        a[1, 0] = corner[0]
        b[1, 0] = corner[1]
        if a[0, 0] == a[1, 0]:
            k = np.zeros((3, 1))
        else:
            k = np.linalg.inv(a).dot(b)
            k = np.concatenate((k, np.array([[1]])), 0)
        ks.append(k)

    # car vis
    a = np.ones((2, 2))
    b = np.zeros((2, 1))
    a[0, 0] = corners[0, 0]
    b[0, 0] = corners[0, 1]
    a[1, 0] = corners[1, 0]
    b[1, 0] = corners[1, 1]
    if a[0, 0] == a[1, 0]:
        k = np.zeros((3, 1))
    else:
        k = np.linalg.inv(a).dot(b)
        k = np.concatenate((k, np.array([[1]])), 0)
    ks.append(k)

    ks = np.stack(ks, 0)
    return ks

def trans_loc_2_area(loc, dims, shape, area_scale):
    loc_area = loc[0:2]*area_scale
    loc_area -=  shape/2
    loc_area = - loc_area
    dims_area = dims[0:2] * area_scale[::-1]
    return loc_area, dims_area

def get_nearst_cars(loc, dims, rot, names, 
                        key_area=[-40, -10, 40, 10],
                        area_scale=[10, 10]):
    car_names = ['CAR', 'TRUCK', 'BUS']
    if len(loc) == 0:
        return []
    loc = np.stack(loc, 0)
    dims = np.stack(dims, 0)
    area = np.zeros(((key_area[2]-key_area[0])*area_scale[0], 
                        (key_area[3]-key_area[1])*area_scale[1]), 
                        dtype=np.bool)
    inds_res = []
    # sort and select target cars
    loc_x = abs(loc[:, 0])
    sort_idx = np.argsort(loc_x)
    area_x = np.arange(area.shape[0])
    area_y = np.arange(area.shape[1])
    area_x_val = np.arange(0, area.shape[0])
    area_axis = np.zeros((area.shape[1], 
                            area.shape[0]))
    area_axis[area_y,:] = np.full((area.shape[0],), 
                                    area_x_val)
    area_axis = np.rot90(area_axis, -1)
    loc_grav = []

    # debug
    # canvas = np.zeros((area.shape[0], 
    #                     area.shape[1], 3))
    # canvas.fill(255)
    # canvas_corners_list = []
    # colors = []

    for idx in sort_idx:
        loc_idx = loc[idx]
        dims_idx = dims[idx]
        rot_idx = rot[idx]
        # real coord to area coord
        loc_area, dims_area = trans_loc_2_area(loc_idx, 
                                                dims_idx,
                                                np.array(area.shape),
                                                area_scale)
        if names[idx] not in car_names or \
            (abs(loc_idx[0]) > key_area[2] or 
                abs(loc_idx[1]) > key_area[3]) :
            continue
        # if area[int(loc_area[0]), 
        #             int(loc_area[1])] :
        #     colors.append((255, 0, 0))
        # else:
        #     colors.append((0, 255, 0))
        # #     continue

        # get corners and transfer to area coord sys
        corners = get_corners(loc_area, 
                                dims_area, rot_idx)
        dis = (corners - 
                np.array(area.shape)/2) ** 2
        dis = np.sum(dis, 1)
        dis_idxs = np.argsort(dis)
        # get max angle corners
        max_ang_corners = get_max_ang_corners(corners, 
                                                loc_area,
                                                np.array(area.shape))
        loc_grav.append([])
        loc_grav[-1].append(corners[dis_idxs[0]]) # naerest corner
        loc_grav[-1].append(corners[dis_idxs[-1]]) # farest corner
        loc_grav[-1].append(np.mean(max_ang_corners, 0))

        # get k by corners
        ks = get_k(max_ang_corners, 
                    np.array(area.shape))
        # get norm & points in area
        inds_list = []
        for i_k, k in enumerate(ks):
            if i_k in [0, 1]:
                loc_j = loc_area
            else:
                loc_j = loc_grav[-1][1]
            if k[2] != 0:
                div = [loc_j[0], loc_j[1]]
                div = [-2 if (_ - area.shape[i_]/2) < 0 else 2
                        for i_, _ in enumerate(div)]
                area_flag = (loc_j[1]+div[1]) > ((loc_j[0]+div[0])*k[0] + k[1])
                area_res = np.zeros_like(area_axis)
                area_res[area_x, :] = np.arange(0, area.shape[1])
                if area_flag:
                    inds = area_res > area_axis*k[0] + k[1]
                else:
                    inds = area_res < area_axis*k[0] + k[1]
            else:
                if i_k in [0, 1]:
                    j_k = i_k
                else:
                    j_k = 0
                area_flag = loc_j[0] > \
                                max_ang_corners[j_k][0]
                inds = np.zeros_like(area)
                if area_flag:
                    inds[int(max_ang_corners[j_k][0]):, :] = 1
                else:
                    inds[:int(max_ang_corners[j_k][0]), :] = 1
            inds_list.append(inds)

        inds_all = np.ones((area.shape[0], 
                                area.shape[1]), 
                                dtype=np.bool)
        for inds in inds_list:
            inds_all = inds_all&inds
        for inds in inds_list:
            inds_all = inds_all&inds
        area[inds_all] = 1

        #canvas_corners_list.append(corners)

    grav_ptr = 0
    for idx in sort_idx:
        loc_idx = loc[idx]
        dims_idx = dims[idx]
        rot_idx = rot[idx]
        if names[idx] not in car_names or \
            (abs(loc_idx[0]) > key_area[2] or 
                abs(loc_idx[1]) > key_area[3]) :
            continue
        loc_area, dims_area = trans_loc_2_area(loc_idx, 
                                                  dims_idx, 
                                                  np.array(area.shape), 
                                                  area_scale)
        loc_j = loc_grav[grav_ptr][-1]
        grav_ptr += 1
        div = np.array([[-2,  2],
                        [-2, -2],
                        [ 2,  2],
                        [ 2, -2]])
        div[:, 0] += int(loc_j[0])
        div[:, 1] += int(loc_j[1])
        inds_res.append(idx)

        # if area[div[:, 0], div[:, 1]].all():
        #     colors.append((255, 0, 0))
        # else:
        #     colors.append((0, 255, 0))

    # cv2.circle(canvas, (int(canvas.shape[0]/2), 
    #                         int(canvas.shape[1]/2)),
    #                         5, (0, 0, 0), -1)
    # add_area(canvas, area)
    # for c_i, corners in enumerate(canvas_corners_list):
    #     add_bbox(canvas, corners, colors[c_i])
    # cv2.imshow('debug', canvas)
    # cv2.waitKey()

    return inds_res

def add_bbox(canvas, corners, color=(0, 255, 0)):
    corners = corners.astype(dtype=np.int)[:, ::-1]
    cv2.polylines(canvas, [corners], True, color, 2)

def add_area(canvas, area):
    canvas[area, :] = [0, 0, 75]

def add_difficulty_to_annos(info, 
                            key_area=np.array([
                                    [[ -10, -5,  10, 5],
                                    [-20, -10, 20, 10]],
                                    [[ -10, -5,  10, 5],
                                    [-20, -10, 20, 10]],

                                    [[ -40, -10,  40, 10],
                                    [-40, -10, 40, 10]],
                                    [[ -40, -10,  40, 10],
                                    [-40, -10, 40, 10]],
                                    [[ -40, -10,  40, 10],
                                    [-40, -10, 40, 10]],
                                    ])):
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # wlh format
    names = annos['name']
    loc = annos['location']
    rot = annos['rotation_y']
    occlusion = annos['occluded']
    truncation = annos['truncated']
    hard_type = annos['hard_type']
    diff = []
    annos['key_area'] = key_area

    key_area_nums = len(key_area[0])
    area_mask = np.ones((len(dims),))

    i = 0
    # TODO remove hardcode keyarea
    cls_idx_map = {
                    'PEDESTRIAN':0,
                    'CYCLIST':1,
                    'CAR':2,
                    'TRUCK':3,
                    'BUS':4,
                }
    # get key car key area0 case
    inds = get_nearst_cars(loc, dims, 
                            rot, names, 
                            key_area[3][0])
    area_mask[inds] = 0
    for i in range(len(loc)):
        if annos['name'][i] not in cls_idx_map:
            cls_idx = 0
        else:
            cls_idx = cls_idx_map[annos['name'][i]]

        if abs(loc[i][0]) <= key_area[cls_idx, 0, 2] and \
                abs(loc[i][1]) <= key_area[cls_idx, 0, 3] and cls_idx in [0, 1]:
            area_mask[i] = 0
        elif abs(loc[i][0]) > key_area[cls_idx, 1, 2] or \
                abs(loc[i][1]) > key_area[cls_idx, 1, 3]:
            area_mask[i] = 2

    for i in range(len(loc)):
        if area_mask[i]==0:
            diff.append(0)
        elif area_mask[i]==1:
            diff.append(1)
        else:
            diff.append(2)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff

def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
