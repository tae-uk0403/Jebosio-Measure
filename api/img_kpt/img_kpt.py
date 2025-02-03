from .convert_json.utils.data_utils import get_affine_transform, normalize, get_gt_class_keypoints_dict
from .convert_json.utils.infer_utils import get_final_preds, transform_preds, _coco_keypoint_results_all_category_kernel
from .convert_json.utils.nms import oks_nms

from .utils.json_to_landmark.text_position_info import position_info
from .utils.json_to_landmark.size_info import getSizingPts
from .utils.json_to_landmark.pyutils import getImageRatio_V2, _kps1d_to_2d, _kps_downscale, get_proj_depth_V2
from .utils.json_to_landmark.pyutils import _kps1d_to_2d, getImageRatio, _kps_downscale, get_proj_depth, get_distance

from plyfile import PlyData
from pyntcloud import PyntCloud
from collections import defaultdict
from pathlib import Path
from PIL import Image

import cv2
import imageio as iio
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import zipfile
import os

# Data Settings
IMAGE_SIZE = [288, 384]
IMG_MEAN, IMG_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
C, S = np.array([959.5, 719.5]), np.array(
    [11.368751, 15.158334])  # center, scale

# post-processing of prediction setiings
target_weight = np.load(Path('api/img_kpt/target_weight.npy'))  # FIXED ???
gt_class_keypoints_dict = get_gt_class_keypoints_dict()  # fixed dictionary
heatmap_height = 96  # = config.MODEL.HEATMAP_SIZE[1]
heatmap_width = 72  # \ config.MODEL.HEATMAP_SIZE[0]

num_samples = 1
NUM_JOINTS = 294
IN_VIS_THRE = 0.2
OKS_THRE = 0.9


def run_img_kpt_processing(task_folder_path: Path,
                           model_version=1,
                           clothes_type=1):

    tflite_model_path = Path("api/img_kpt/model/test_hrnet.tflite")
    # 1 : 반팔 , 7 : 반바지 , 8 : 긴바지(pants)
    img_file = task_folder_path / 'image_file.jpg'
    res_file = task_folder_path / 'keypoint.json'

    # Model(TFLite) Load
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_model_path.absolute()))
    interpreter.allocate_tensors()

    # Get input and output details(including the shape)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # input_shape = input_details[0]['shape']

    # Data load and transform
    data_numpy = cv2.imread(str(img_file), cv2.IMREAD_COLOR |
                            cv2.IMREAD_IGNORE_ORIENTATION)
    c, s, r = C, S, 0
    trans = get_affine_transform(c, s, r, IMAGE_SIZE)
    input_data = cv2.warpAffine(data_numpy, trans,
                                (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
                                flags=cv2.INTER_LINEAR)
    input_data = normalize(input_data, IMG_MEAN, IMG_STD)
    input_data = np.array(input_data, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0).transpose(0, 3, 1, 2)
    print('input_data.shape : ', input_data.shape)

    # Model input and output
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # for final-result
    all_preds = np.zeros((num_samples, NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 7))

    # output to json format
    idx = 0

    num_images = 1  # one image
    cat_ids = [clothes_type]  # 반팔=1 ,
    c = np.expand_dims(C, 0)
    s = np.expand_dims(S, 0)
    score = [1]  # fixed

    channel_mask = np.zeros_like(target_weight, dtype=np.float32)
    rg = gt_class_keypoints_dict[int(cat_ids[0])]
    index = np.array([list(range(rg[0], rg[1]))],
                     dtype=np.int32).transpose(1, 0)
    # like pytorch _scatter -> ex) channel_mask[j].scatter_(0, index, 1)
    for i in index:
        channel_mask[0][i] = [1]

    mask = np.expand_dims(channel_mask, axis=3)
    output = output * mask

    preds_local, maxvals = get_final_preds(
        heatmap_height, heatmap_width, output, c, s)
    preds = preds_local.copy()
    for i in range(preds_local.shape[0]):
        preds[i] = transform_preds(
            preds_local[i], c[i], s[i],
            [heatmap_width, heatmap_height])

    all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]  # from Pred
    all_preds[idx:idx + num_images, :, 2:3] = maxvals  # from Pred

    # double check this all_boxes parts
    all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
    all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
    all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
    all_boxes[idx:idx + num_images, 5] = score
    all_boxes[idx:idx + num_images, 6] = np.array(cat_ids)

    _kpts = []
    for idx, kpt in enumerate(all_preds):  # preds (2, 294, 3)
        _kpts.append({
            'keypoints': kpt,  # (294, 3)
            'center': all_boxes[idx][0:2],
            'scale': all_boxes[idx][2:4],
            'area': all_boxes[idx][4],
            'score': all_boxes[idx][5],
            'category_id': all_boxes[idx][6],
            #         'image': int(img_path[idx][-10:-4]) # *** acceptable only for last 6-digit
            'image': 1
        })

    kpts = defaultdict(list)
    for kpt in _kpts:
        kpts[kpt['image']].append(kpt)

    oks_nmsed_kpts = []
    for img in kpts.keys():  # each image
        img_kpts = kpts[img]
        for n_p in img_kpts:  # each items
            box_score = n_p['score']  # score from 'meta'
            kpt_score = 0
            valid_num = 0
            for n_jt in range(0, NUM_JOINTS):  # rach jt id
                t_s = n_p['keypoints'][n_jt][2]  # 3rd value
                if t_s > IN_VIS_THRE:  # if t_s > 0.2
                    kpt_score = kpt_score + t_s
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            n_p['score'] = kpt_score * box_score

        keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], OKS_THRE)
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    # json write
    results = _coco_keypoint_results_all_category_kernel(
        oks_nmsed_kpts, NUM_JOINTS)
    with open(res_file, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    print(
        f'Successfully saved the model result for the RGB image as a json file ! ---> {res_file} ')

    if model_version == 1:
        _landmark_task_1(task_folder_path, clothes_type)
    elif model_version == 2:
        _landmark_task_2(task_folder_path, clothes_type)

    zipFileName = "result.zip"
    zf = zipfile.ZipFile(os.path.join(task_folder_path, zipFileName), "w")
    # # 결과 파일을 압축합니다.
    zf.write(os.path.join(task_folder_path, 'result_image_v1.png'))
    zf.write(os.path.join(task_folder_path, 'result_kpt_v1.json'))
    zf.close()
    return 


''' hyper-params '''
deg = 0.275


def _landmark_task_1(task_folder_path, clothes_type):
    root = task_folder_path
    print('root :', root)

    ''' File Load '''
    predicted_json_file = task_folder_path / 'keypoint.json'
    print('predicted_json_file : ', predicted_json_file)
    with predicted_json_file.open('r') as f:
        predicted_json = json.load(f)

    img_path = task_folder_path / 'image_file.jpg'
    ply_path = task_folder_path / 'ply_file.ply'
    print('img_path :', img_path)
    print('ply_path :', ply_path)

    image = Image.open(img_path)
    ply = PlyData.read(ply_path)
    print('image size :', image.size)
    ''' '''

    ''' json to Landmark position & length(cm) calculation '''
    measure_index, measure_points = getSizingPts(clothes_type)  # 1: 반팔
    w_r, h_r = getImageRatio(image, ply)

    predicted = predicted_json[0]
    pred_kpt1d = predicted['keypoints']
    kps_arr = _kps1d_to_2d(pred_kpt1d)
    kps_arr = _kps_downscale(kps_arr, (w_r, h_r))
    kps_dict = {i + 1: arr for i, arr in enumerate(kps_arr)}
    ply_dpt = get_proj_depth(ply)  # projected depth array

    result = {name: {'pt1': None, 'pt2': None, 'cm': None}
              for name in measure_index.values()}
    for a, name in measure_index.items():
        pt1, pt2 = [tuple(kps_dict[pt_key].astype(int))
                    for pt_key in measure_points[a]]
        result[name]['depth_pt1'] = int(pt1[0]), int(pt1[1])
        result[name]['depth_pt2'] = int(pt2[0]), int(pt2[1])
        result[name]['pt1'] = int(pt1[0] * w_r), int(pt1[1] * h_r)
        result[name]['pt2'] = int(pt2[0] * w_r), int(pt2[1] * h_r)

        size_ = get_distance(pt1, pt2, ply_dpt, deg=deg)
        result[name]['cm'] = round(size_, 2)

    ''' '''

    ''' Draw a circle, line and length on an image '''
    img_arr = np.array(image)
    

    text_pos = position_info(clothes_type)


    for r in result:
        pt1, pt2 = result[r]['pt1'], result[r]['pt2']
        cm = result[r]['cm']

        for c in [pt1, pt2]:
            cv2.circle(img_arr,
                       c,
                       10,
                       (255, 0, 0),
                       thickness=-1
                       )
        cv2.line(img_arr,
                 pt1,
                 pt2,
                 (255, 0, 0),
                 thickness=4,
                 lineType=cv2.LINE_AA)

        if text_pos != 0:
            cv2.putText(img_arr, r, (pt2[0] + text_pos['Main'][0], pt2[1] + text_pos['Main'][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 6)
            cv2.putText(img_arr, f'({cm}cm)', (pt2[0] + text_pos[r][0], pt2[1] + text_pos[r][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 6)


    
    print(result)
    with open(task_folder_path / 'result_kpt_v1.json', 'w') as f:
        json.dump(result, f)

    result_img = Image.fromarray(img_arr)
    result_img.save(task_folder_path / 'result_image_v1.png', 'png')

    print('Finish !')
    print(f'Result json save --> {task_folder_path / "result_kpt_v1.png"}')
    print(f'Result Image save --> {task_folder_path / "result_image.png"}')


''' hyper-params '''
FX_DEPTH = 5.8262448167737955e+02
FY_DEPTH = 5.8269103270988637e+02
CX_DEPTH = 3.1304475870804731e+02
CY_DEPTH = 2.3844389626620386e+02


def _landmark_task_2(task_folder_path, clothes_type):
    root = task_folder_path
    print('root :', root)

    ''' File Load '''
    predicted_json_file = task_folder_path / 'keypoint.json'
    print('predicted_json_file : ', predicted_json_file)
    with predicted_json_file.open('r') as f:
        predicted_json = json.load(f)

    img_path = task_folder_path / 'image_file.jpg'
    depth_path = task_folder_path / 'depth_file.jpg'
    image = Image.open(img_path)
    depth_image = iio.imread(depth_path)
    depth_image = depth_image[:, :, 0]
    # depth_image = depth_image.transpose(1, 0) # 20230209 hm landmark_v2에서 업데이트 후 롤백
    print('img_path :', img_path)
    print('depth_path :', depth_path)
    print('image shape : ', image.size)
    print('depth_image shape : ', depth_image.shape)  # (192, 256)
    ''' '''

    ''' Make new_ply.ply & Load new_ply '''
    img_arr = np.asarray(image)
    pcd = []
    height, width = depth_image.shape  # (192, 256)
    for i in range(height):
        for j in range(width):
            z = depth_image[i][j]
            x = (j - CX_DEPTH) * z / FX_DEPTH
            y = (i - CY_DEPTH) * z / FY_DEPTH
            r = img_arr[:, :, 0][i][j]
            g = img_arr[:, :, 1][i][j]
            b = img_arr[:, :, 2][i][j]
            x_coord, y_coord = j, i
            pcd.append([x, y, z, r, g, b, x_coord, y_coord])

    pcd = np.array(pcd, dtype=np.float32)
    pcd = {'x': pcd[:, 0],
           'y': pcd[:, 1],
           'z': pcd[:, 2],
           'r': pcd[:, 3],
           'g': pcd[:, 4],
           'b': pcd[:, 5],
           'x_coord': pcd[:, 6],
           'y_coord': pcd[:, 7]
           }
    # build a cloud
    cloud = PyntCloud(pd.DataFrame(pcd))
    #cloud.to_file(root / 'new_ply.ply', as_text=True)
    cloud.to_file(str(root / 'new_ply.ply'), as_text=True)

    ply = PlyData.read(root / 'new_ply.ply')
    assert ply['vertex'].count == 49152
    ''' '''

    ''' json to Landmark position & length(cm) calculation '''
    measure_index, measure_points = getSizingPts(clothes_type)  # 1: 반팔
    w_r, h_r = getImageRatio_V2(image, depth_image.shape)
    print(f'measure index is {measure_index}')
    print(f'measure point is {measure_points}')

    predicted = predicted_json[0]
    pred_kpt1d = predicted['keypoints']
    kps_arr = _kps1d_to_2d(pred_kpt1d)
    kps_arr = _kps_downscale(kps_arr, (w_r, h_r))
    kps_dict = {i + 1: arr for i, arr in enumerate(kps_arr)}
    ply_dpt = get_proj_depth_V2(
        ply, depth_image.shape)  # projected depth array

    result = {name: {'pt1': None, 'pt2': None,
                     'depth_pt1': None, 'depth_pt2': None,
                     'cm': None} for name in measure_index.values()}
    for a, name in measure_index.items():
        pt1, pt2 = [tuple(kps_dict[pt_key].astype(int))
                    for pt_key in measure_points[a]]
        result[name]['depth_pt1'] = int(pt1[0]), int(pt1[1])
        # result[name]['depth_pt2'] = int(pt1[0]), int(pt1[1]) # 20230209 hm landmark_v2에서 업데이트 후 롤백
        result[name]['depth_pt2'] = int(pt2[0]), int(pt2[1])
        result[name]['pt1'] = int(pt1[0] * w_r), int(pt1[1] * h_r)
        result[name]['pt2'] = int(pt2[0] * w_r), int(pt2[1] * h_r)

        point_a = ply_dpt[pt1]
        point_b = ply_dpt[pt2]
        size_ = np.linalg.norm(point_a - point_b)
        result[name]['cm'] = float(str(size_)[:5])

    ''' '''

    ''' Draw a circle, line and length on an image '''
    img_arr = np.asarray(image)
    if not img_arr.flags.writeable:
        img_arr = img_arr.copy()
    text_pos = position_info(clothes_type)

    for r in result:
        pt1, pt2 = result[r]['pt1'], result[r]['pt2']
        cm = result[r]['cm']

        for c in [pt1, pt2]:
            cv2.circle(img_arr,
                       c,
                       5,
                       (255, 0, 0),
                       thickness=-1
                       )
        cv2.line(img_arr,
                 pt1,
                 pt2,
                 (255, 0, 0),
                 thickness=2,
                 lineType=cv2.LINE_AA)
        if r=='Total-length':
            cv2.putText(img_arr, r, (pt1[0] + text_pos['Main'][0], pt1[1] + text_pos['Main'][1]), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 6)
            cv2.putText(img_arr, f'({cm}cm)', (pt1[0] + text_pos[r][0], pt1[1] + text_pos[r][1]), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 6)
        else:
            cv2.putText(img_arr, r, (pt2[0] + text_pos['Main'][0], pt2[1] + text_pos['Main'][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 6)
            cv2.putText(img_arr, f'({cm}cm)', (pt2[0] + text_pos[r][0], pt2[1] + text_pos[r][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 6)

    ''' result Image and json save'''
    print(result)
    with open(root / 'result_kpt_v1.json', 'w') as f:
        json.dump(result, f)

    result_img = Image.fromarray(img_arr)
    result_img.save(root / 'result_image_v1.png', 'png')


if __name__ == '__main__':
    run_img_kpt_processing()
