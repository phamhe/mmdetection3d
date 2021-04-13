import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io
import json

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

def add_difficulty_to_annos(info, 
                            key_area=[
                                    [ -3, -10,  3, 10],
                                    [-10, -20, 10, 20],
                                     ]):
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    loc = annos['location']
    occlusion = annos['occluded']
    truncation = annos['truncated']
    hard_type = annos['hard_type']
    diff = []
    annos['key_area'] = key_area

    key_area_nums = len(key_area[0])
    area_mask = np.zeros((len(dims),))

    i = 0
    #TODO remove hardcode keyarea
    for i in range(len(loc)):
        if (abs(loc[i][0]) <= key_area[0][2] and 
                abs(loc[i][1]) <= key_area[0][3]):
            continue
        elif abs(loc[i][0]) <= key_area[1][2] or \
                abs(loc[i][1]) <= key_area[1][3]:
            area_mask[i] = 1
        else:
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
