import copy
import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_result, show_results, show_results_bev
from ..core.bbox import (Box3DMode, LiDARInstance3DBoxes, Coord3DMode)
from .custom_3d import Custom3DDataset

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline


@DATASETS.register_module()
class DeeprouteDataset(Custom3DDataset):
    r"""Deeproute Dataset.

    This class serves as the API for experiments on the `Deeproute Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-20, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ('PEDESTRIAN', 'CYCLIST', 'CAR', 'TRUCK', 'BUS')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=[-74.88, -74.88, -4, 74.88, 74.88, 4]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix
        self.cls_name = {
                    0: 'ped',
                    1: 'cyc',
                    2: 'car', 3: 'truck', 4: 'bus',
                    }
        self.cls_idx = {
                    'pedestrian': 0,
                    'cyclist': 1,
                    'car': 2, 'truck': 3, 'bus': 4,
                    }

    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:05d}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_info (dict): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']

        # TODO: consider use torch.Tensor only

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1])

        # debug code
        # example = self.prepare_test_data(index)
        # data_info = self.data_infos[index]
        # pts_path = data_info['point_cloud']['velodyne_path']
        # file_name = osp.split(pts_path)[-1].split('.')[0]
        # points = example['points'][0]._data.numpy()
        # show_result(points, 
        #             gt_bboxes_3d.tensor.numpy(), 
        #             gt_bboxes_3d.tensor.numpy(), 
        #             'model_0324_deeproute_pp', file_name, True)
        # exit()

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if 'pts_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_deeproute(results_, self.CLASSES,
                                                       pklfile_prefix_,
                                                       submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_deeproute(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)

        return result_files, tmp_dir
    def plot_extra(self, x, y, prefix, postfix, out_dir):
        color = {
                0:['red', 'darkred', 'lightcoral'],
                1:['green', 'darkgreen', 'lightgreen'],
                2:['blue', 'darkblue', 'royalblue'],
                # 3:['cyan', 'darkcyan', 'darkslategray'],
                # 4:['']
                }
        cls_name = {
                    0: 'ped',
                    1: 'cyc',
                    2: 'car', 3: 'truck', 4: 'bus',
                    }
        iou_name = [
                    [0.3, 0.5],
                    [0.3, 0.5],
                    [0.5, 0.7],
                    [0.5, 0.7],
                    [0.5, 0.7]
                    ]
        plt_ptr = 0
        for i_cls in range(x.shape[0]):
            plt_ptr += 1
            plt.figure(plt_ptr)
            for i_ka in range(x.shape[1]):
                for i_iou in range(x.shape[2]):
                    fx = x[i_cls, i_ka, i_iou, :]
                    fy = y[i_cls, i_ka, i_iou, :]
                    inds = fx != 0
                    fx = fx[inds]
                    fy = fy[inds]

                    # fx, inds = np.unique(fx, return_index=True)
                    # fy = fy[inds]
                    # fx.sort()
                    # fx = fx[::-1]
                    if not fx.any() or fx.shape[0]<3:
                        continue
                    # li = interp1d(fx, fy, kind='quadratic')
                    # fx_new = np.linspace(fx[0], fx[-1], 200)
                    # fy_new = li(fx_new)
                    # li = make_interp_spline(fx[::-1], fy[::-1])
                    # fy_new = li(fx_new)

                    label='%s_ka%s_iou%s_%s'%(cls_name[i_cls], 
                                               i_ka, iou_name[i_cls][i_iou], 
                                               postfix)
                    title='%s_iou%s_%s'%(cls_name[i_cls], 
                                               iou_name[i_cls][i_iou], 
                                               postfix)
                    # plt.plot(fx_new, fy_new, 
                    #          color=color[i_ka][i_iou],
                    #          label=label) 
                    plt.plot(fx, fy, 
                             color=color[i_ka][i_iou],
                             label=label) 
                    plt.xlabel(prefix)
                    plt.ylabel('times')
                    plt.title(title)
                    plt.legend()
            name ='%s_%s_%s'%(cls_name[i_cls], prefix, postfix)
            plt.savefig(os.path.join(out_dir, name))
            plt.close()
                    
    # def plot_extra(self, eval_res, out_dir, cls_group=[[0, 1], [2, 3, 4]]):
    #     color = {
    #             0:['red', 'darkred', 'lightcoral'],
    #             1:['green', 'darkgreen', 'lightgreen'],
    #             2:['blue', 'darkblue', 'royalblue'],
    #             }
    #     plt_ptr = 0
    #     group_ptr = 0
    #     for level in eval_res:
    #         eval_res_level = eval_res[level]
    #         for eval_key in eval_res_level:
    #             for idx, cls_err_all in enumerate(eval_res_level[eval_key]):
    #                 if idx not in cls_group[group_ptr]:
    #                     group_ptr += 1
    #                     while group_ptr > len(cls_group) - 1:
    #                         group_ptr %= len(cls_group)
    #                     plt_ptr += 1
    #                     plt.figure(plt_ptr)

    #                 cls_err_all = np.stack(cls_err_all, axis=0)
    #                 cls_err_all = np.sort(cls_err_all, 0)
    #                 cls_err_min_all = cls_err_all[0]
    #                 cls_err_max_all = cls_err_all[-1]
    #                 axis_x_all = np.linspace(cls_err_min_all, cls_err_max_all, 60)
    #                 if len(axis_x_all.shape) > 1:
    #                     dim = axis_x_all.shape[1]
    #                 else:
    #                     dim = 1
    #                 for i in range(dim):
    #                     cls_err = cls_err_all
    #                     axis_x = axis_x_all
    #                     cls_err_min = cls_err_min_all
    #                     cls_err_max = cls_err_max_all
    #                     if len(axis_x_all.shape) > 1:
    #                         cls_err = cls_err_all[:, i]
    #                         axis_x = axis_x_all[:, i]
    #                         cls_err_min = cls_err_min_all[i]
    #                         cls_err_max = cls_err_max_all[i]
    #                     axis_y = np.zeros(axis_x.shape[0])

    #                     ptr = 0
    #                     count = 0
    #                     for j, err in enumerate(cls_err):
    #                         count += 1
    #                         if err >= axis_x[ptr]:
    #                             axis_y[ptr] = count
    #                             ptr += 1
    #                             count = 0

    #                     li = interp1d(axis_x, axis_y, kind='quadratic')
    #                     axis_x_new = np.linspace(cls_err_min, cls_err_max, 1000)
    #                     axis_y_new = li(axis_x_new)
    #                     plt.plot(axis_x_new, axis_y_new, 
    #                             color=color[idx%(
    #                                     len(cls_group[group_ptr]))][i],
    #                              label='cls_%s_%s_%s_%s'%(idx, level, eval_key, i))
    #                     # plt.plot(axis_x, axis_y, 'o',
    #                     #         color=color[idx%(
    #                     #                 len(cls_group[group_ptr]))][i],
    #                     #          label='cls_%s_%s_%s_%s'%(idx, level, eval_key, i))
    #                 plt.xlabel(eval_key)
    #                 plt.ylabel('times')
    #                 plt.title('cls_%s_%s_%s'%(idx, level, eval_key))
    #                 plt.legend()
    #                 # plt.show()
    #                 plt.savefig(os.path.join(out_dir, 'cls_%s_%s_%s'%(idx, level, eval_key)))
    #                 plt.close()

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import deeproute_eval
        gt_annos = [info['annos'] for info in self.data_infos]

        ap_dict = dict()
        if 'pts_bbox' in result_files:
            result_files_ = result_files['pts_bbox']
            eval_types = ['bev', '3d']
            ap_result_str, ap_dict_, curve_res_bev, curve_res_3d, extra_res = deeproute_eval(
                gt_annos,
                result_files_,
                self.CLASSES,
                eval_types=eval_types)

            for ap_type, ap in ap_dict_.items():
                ap_dict[f'{ap_type}'] = float('{:.4f}'.format(ap))

            print_log(
                f'Results :\n' + ap_result_str, logger=logger)
        if tmp_dir is not None:
            tmp_dir.cleanup()

        info = [extra_res, curve_res_3d, curve_res_bev]
        return info

    def coarl_anly(self, curve_res_3d, out_dir):
        self.plot_extra(curve_res_3d['recall'], 
                        curve_res_3d['fp'], 
                        'recall', 'fp_3d', 
                        out_dir)
        self.plot_extra(curve_res_3d['recall'], 
                        curve_res_3d['fn'], 
                        'recall', 'fn_3d', 
                        out_dir)
        self.plot_extra(curve_res_3d['thresh'], 
                        curve_res_3d['precision'], 
                        'thresh', 'precision', 
                        out_dir)
        self.plot_extra(curve_res_3d['thresh'], 
                        curve_res_3d['recall'], 
                        'thresh', 'recall', 
                        out_dir)

    def fine_anly(self, curve_res_3d, show_inds):
        res_str = ''
        for cls in range(show_inds.shape[0]):
            res_str += '-----------------------------\n'
            res_str += '{} results\n'.format(self.cls_name[cls])
            for ka in range(show_inds.shape[1]):
                for iou in range(show_inds.shape[2]):
                    res_str += 'key_area{}, min_iou{}\n'.format(ka, iou)
                    keys = list(curve_res_3d.keys())
                    strs = ['{0:9} :'.format(key) for key in keys]
                    for i_str, str_ in enumerate(strs):
                        key = keys[i_str]
                        case = curve_res_3d[key][cls, ka, iou]
                        case = case[show_inds[cls, ka, iou]]
                        for ca in case:
                            strs[i_str] += ' {0:7},'.format(str('%.4f'%ca))
                    for str_ in strs:
                        str_ += '\n'
                        res_str += str_
        return res_str

    def online_eval(self, results, info, out_dir):
        extra_res = info[0]
        curve_res_3d = info[1]
        self.coarl_anly(curve_res_3d, out_dir)
        show_inds = self.get_thresh_inds(extra_res, curve_res_3d)
        res_str = self.fine_anly(curve_res_3d, show_inds)
        print(res_str)
        fresults, fps, fns = self.get_fine_res(results, extra_res)
        ffps_idx, ffns_idx = self.get_final_index(fps, fns)
        self.show_badcase(results, out_dir, fresults, ffps_idx, ffns_idx)

    def show_badcase(self, results, out_dir, fresults, ffps_idx, ffns_idx):
        for cls in range(len(ffps_idx)):
            for diff in range(len(ffps_idx[cls])):
                if not len(ffps_idx[cls][diff]):
                    continue
                fps_idx = ffps_idx[cls][diff][0]
                fns_idx = ffns_idx[cls][diff][0]
                for i in fps_idx:
                    self.show(i, results, out_dir, fresults)
                for i in fns_idx:
                    self.show(i, results, out_dir, fresults)

    def get_final_index(self, fps, fns,
                        classes=[2],
                        difficulty=[0],
                        thresh=1.0):
        
        ffps_idx = []
        ffns_idx = []
        for i in range(fps.shape[0]):
            ffps_idx.append([])
            ffns_idx.append([])
            for j in range(fps.shape[2]):
                ffps_idx[i].append([])
                ffns_idx[i].append([])

        for diff in difficulty:
            for cls in classes:
                fps_cls = fps[cls, :, diff]
                fps_num = fps_cls.sum()
                fns_cls = fns[cls, :, diff]
                fns_num = fns_cls.sum()

                fps_num_sel = int(fps_num*thresh)
                fns_num_sel = int(fns_num*thresh)

                args_fps = np.argsort(fps_cls)[::-1]
                args_fns = np.argsort(fns_cls)[::-1]

                fps_sum = 0
                fns_sum = 0
                fps_ptr = 0
                fns_ptr = 0
                for i_fps in args_fps:
                    fps_sum += fps_cls[i_fps]
                    fps_ptr += 1
                    if fps_sum >= fps_num_sel:
                        break
                for i_fns in args_fns:
                    fns_sum += fns_cls[i_fns]
                    fns_ptr += 1
                    if fns_sum >= fns_num_sel:
                        break

                fps_idx = args_fps[:fps_ptr+1]
                ffps_idx[cls][diff].append(fps_idx)
                fns_idx = args_fns[:fns_ptr+1]
                ffns_idx[cls][diff].append(fns_idx)
        return ffps_idx, ffns_idx

    def get_thresh_inds(self, extra_res, curve_res_3d,
                      recall_thresh_rel=np.array([0.9, 0.92, 0.94, 0.96, 0.98, 1.0]),
                      key_area = [0],
                      iou_thresh = [0]
                      ):

        thresholds = curve_res_3d['thresh']
        recalls = curve_res_3d['recall']
        tps = curve_res_3d['tp']
        fps = curve_res_3d['fp']
        fns = curve_res_3d['fn']

        # get indexs
        indexs = np.zeros((tps.shape[0], len(key_area), len(iou_thresh), thresholds.shape[3]), dtype=np.bool)

        for i_cls in range(recalls.shape[0]):
            for i_ka in key_area:
                for idx, i_iou in enumerate(iou_thresh):
                    inds = recalls[i_cls][i_ka][i_iou] != 0
                    recalls_cls = recalls[i_cls][i_ka][i_iou][inds]
                    recall_thresh = [_*recalls_cls[-1] for _ in recall_thresh_rel]
                    idxs_list = np.zeros((recalls.shape[3],), dtype=np.bool)
                    thr_ptr = 0
                    for i_recall in range(recalls_cls.shape[-1]):
                        if thr_ptr >= len(recall_thresh):
                            break
                        elif recalls_cls[i_recall] >= recall_thresh[thr_ptr]:
                            idxs_list[i_recall] = 1
                            thr_ptr += 1
                    indexs[i_cls, i_ka, i_iou][:] = idxs_list
        return indexs

    def bbox2result_deeproute(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None,
                          key_area=[
                                    [20, 40],
                                    [10, 20]
                                    ]):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to Deeproute format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
                'difficulty':[],
            }
            if len(box_dict['box3d_lidar']) > 0:
                box_preds = box_dict['box3d_lidar']
                scores = box_dict['scores']
                label_preds = box_dict['label_preds']

                for box, score, label in zip(
                        box_preds, scores, label_preds):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)
                    if (abs(box[0]) <= key_area[0][0] and
                            abs(box[1]) <= key_area[1][0]) or \
                            (abs(box[0]) <= key_area[0][0] and 
                            abs(box[1]) <= key_area[1][1]) or \
                            (abs(box[0]) <= key_area[0][1] and
                            abs(box[1]) <= key_area[1][0]):
                        anno['difficulty'].append(0)
                    elif abs(box[0]) <= key_area[0][1] or \
                            abs(box[1]) <= key_area[1][1]:
                        anno['difficulty'].append(1)
                    else:
                        anno['difficulty'].append(2)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        # box_preds.tensor[:, -1] = box_preds.tensor[:, -1] - np.pi
        # box_preds.limit_yaw(offset=0, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

    def get_fine_res(self, results, extra_res, 
                thresh=np.array([0.31, 0.36, 0.75, 0.52, 0.77]),
                iou=np.array([0.3, 0.3, 0.5, 0.5, 0.5])):

        # get fps && fns in score & iou
        fps = np.zeros((len(self.cls_name), len(results), 3))
        fns = np.zeros((len(self.cls_name), len(results), 3))
        final_results = []
        for i, result in enumerate(results):
            frame_extra = extra_res[i]
            gts = frame_extra['gts']
            dts = frame_extra['dts']
            ious = frame_extra['ious']
            max_ious = np.zeros(ious.shape[0])
            inds = np.zeros(ious.shape[0])
            ious_flag = False
            if ious.any():
                ious_flag = True
                max_ious, inds = torch.tensor(ious).max(dim=1)
            # set inds
            inds_set = {}
            inds_table = set()
            for dt_idx, gt_idx in enumerate(inds):
                if int(gt_idx) not in inds_set:
                    inds_set[int(gt_idx)] = []
                inds_set[int(gt_idx)].append(dt_idx)
                inds_table.add(dt_idx)
            frame_res = []
            for idx, cls in enumerate(gts['name']):
                if cls.lower() not in self.cls_idx:
                    continue
                res = {
                        'gt_annos': {
                                    'name':gts['name'][idx],
                                    'dimensions':gts['dimensions'][idx],
                                    'location':gts['location'][idx],
                                    'rotation_y':np.array([gts['rotation_y'][idx]]),
                                    'difficulty':np.array([gts['difficulty'][idx]]),
                                    'fn_flag':False,
                                    },
                        'dts':[],
                        'fps':[],
                        'fns':[],
                        'others':[],
                        }
                gt_name = gts['name'][idx].lower()
                if not ious_flag:
                    for j in range(dts['name'].shape[0]):
                        dt_name = dts['name'][j].lower()
                        dt = {
                             'name':dts['name'][j],
                             'dimensions':dts['dimensions'][j],
                             'location':dts['location'][j],
                             'rotation_y':np.array([dts['rotation_y'][j]]),
                             'score':np.array([dts['score'][j]]),
                             'iou':np.array([0]),
                             'difficulty':np.array([dts['difficulty'][j]])
                            }
                        if dts['score'][j] >= thresh[self.cls_idx[dt_name]]:
                            res['fps'].append(dt)
                            fps[self.cls_idx[dt_name], i, dts['difficulty'][j]] += 1
                    fn = {
                         'name':gts['name'][idx],
                         'dimensions':gts['dimensions'][idx],
                         'location':gts['location'][idx],
                         'rotation_y':np.array([gts['rotation_y'][idx]]),
                         'difficulty':np.array([gts['difficulty'][idx]])
                        }
                    res['fns'].append(fn)
                    fns[self.cls_idx[gt_name], i, gts['difficulty'][idx]] += 1
                    res['gt_annos']['fn_flag'] = True
                else:
                    fn = {
                         'name':gts['name'][idx],
                         'dimensions':gts['dimensions'][idx],
                         'location':gts['location'][idx],
                         'rotation_y':np.array([gts['rotation_y'][idx]]),
                         'difficulty':np.array([gts['difficulty'][idx]]),
                         'score':np.array([0.0]),
                         'iou':np.array([0.0])
                        }
                    # recall
                    if idx in inds_set:
                        for j in inds_set[idx]:
                            dt_name = dts['name'][j].lower()
                            dt = {
                                 'name':dts['name'][j],
                                 'dimensions':dts['dimensions'][j],
                                 'location':dts['location'][j],
                                 'rotation_y':np.array([dts['rotation_y'][j]]),
                                 'score':np.array([dts['score'][j]]),
                                 'iou':np.array([max_ious[j]]),
                                 'difficulty':np.array([dts['difficulty'][j]])
                                }
                            if max_ious[j] >= iou[self.cls_idx[dt_name]] and \
                                gts['name'][idx] == dts['name'][j] and \
                                dts['score'][j] >= thresh[self.cls_idx[dt_name]]:
                                res['dts'].append(dt)
                            elif dts['score'][j] >= thresh[self.cls_idx[dt_name]]:
                                res['fps'].append(dt)
                                fps[self.cls_idx[dt_name], i, dts['difficulty'][j]] += 1
                            # has iou with gt
                            elif max_ious[j] > 0:
                                res['others'].append(dt)
                            if j in inds_table:
                                inds_table.discard(j)
                        if len(res['dts']) == 0:
                            res['fns'].append(fn)
                            fns[self.cls_idx[gt_name], i, gts['difficulty'][idx]] += 1
                            res['gt_annos']['fn_flag'] = True
                    else:
                        res['fns'].append(fn)
                        res['gt_annos']['fn_flag'] = True
                        fns[self.cls_idx[gt_name], i, gts['difficulty'][idx]] += 1
                frame_res.append(res)
            for j in inds_table:
                fp = {
                     'name':dts['name'][j],
                     'dimensions':dts['dimensions'][j],
                     'location':dts['location'][j],
                     'rotation_y':np.array([dts['rotation_y'][j]]),
                     'score':np.array([dts['score'][j]]),
                     'iou':np.array([max_ious[j]]),
                     'difficulty':np.array([dts['difficulty'][j]])
                    }
                if dts['score'][j] >= thresh[self.cls_idx[dt_name]]:
                    if len(frame_res) > 0:
                        frame_res[-1]['fps'].append(fp)
                    else:
                        frame_res.append(
                                        {
                                        'gt_annos':None,
                                        'dts':[],
                                        'fps':[fp],
                                        'fns':[]
                                        }
                                        )
                    fps[self.cls_idx[dt_name], i, dts['difficulty'][j]] += 1
            final_results.append(frame_res)
        return final_results, fps, fns

    def show(self, i, results, out_dir, extra_res,
                show=True):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
        """
        color_map = {
                    'gt_annos':(0, 255, 0),
                    'dts':(0, 0, 255),
                    'fns':(255, 153, 0),
                    'fps':(255, 0, 0),
                    'others':(0, 153, 255),
                    }

        result = results[i]
        example = self.prepare_test_data(i)
        data_info = self.data_infos[i]
        pts_path = data_info['point_cloud']['velodyne_path']
        file_name = osp.split(pts_path)[-1].split('.')[0]
        points = example['points'][0]._data.numpy()
        gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor

        # frame in res
        if 'pts_bbox' in result:
            result = result['pts_bbox']
        pred_bboxes = result['boxes_3d'].tensor.numpy()
        # show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name,
        #             show)

        # frame in extra_res
        frame_extra = extra_res[i]
        allbboxes = []
        colors = []
        labels = []
        keys = []
        for key in frame_extra[0].keys():
            if key not in [
                            'gt_annos',
                            'dts',
                            'fps',
                            'fns',
                            'others',
                            ]:
                continue
            keys.append(key)
        bboxes_dict = {key:np.zeros((1, 7)) for key in keys}
        labels_dict = {key:np.zeros((1, 7)) for key in keys} # cls, cls_idx, score, ious, difficulty, location

        # res in frame
        for res in frame_extra:
            for key in keys:
                infos = res[key]
                if isinstance(infos, list):
                    for info in infos:
                        bbox3d = np.concatenate((info['location'],
                                                    info['dimensions'],
                                                    info['rotation_y']))
                        bbox3d = bbox3d.reshape(1, 7)
                        label = np.concatenate((np.array([info['name']]),
                                                np.array([self.cls_idx[info['name'].lower()]]),
                                                info['score'],
                                                info['iou'],
                                                info['difficulty'],
                                                info['location'][:-1]))
                        label = label.reshape(1, 7)
                        bboxes_dict[key] = np.concatenate((bboxes_dict[key], bbox3d), 0)
                        labels_dict[key] = np.concatenate((labels_dict[key], label), 0)
                # gt
                else:
                    if infos['fn_flag']:
                        continue
                    bbox3d = np.concatenate((infos['location'],
                                                infos['dimensions'],
                                                infos['rotation_y']))
                    bbox3d = bbox3d.reshape((1, 7))
                    label = np.concatenate((np.array([infos['name']]),
                                            np.array([self.cls_idx[infos['name'].lower()]]),
                                            np.array([1.0]),
                                            np.array([0.0]),
                                            infos['difficulty'],
                                            infos['location'][:-1]))
                    label = label.reshape((1, 7))
                    bboxes_dict[key] = np.concatenate((bboxes_dict[key], bbox3d), 0)
                    labels_dict[key] = np.concatenate((labels_dict[key], label), 0)
        for key in keys:
            allbboxes.append(bboxes_dict[key][1:])
            colors.append(color_map[key])
            labels.append(labels_dict[key][1:])
        pcd_limit_range = [-40, -20, 0, 40, 20, 0]
        show_results_bev(points, allbboxes, colors, keys, pcd_limit_range, out_dir, i, labels)
        show_results(points, allbboxes, colors, out_dir, file_name, show)
