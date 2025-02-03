import cv2
import torch 
import os
import torchvision.transforms as transforms
from .utils import pose_hrnet
from .utils.transforms import get_affine_transform, transform_preds
from .utils.class_dict import *
from .utils.inference import *
from .utils.functions import *
from .utils.bbox_finder import bbox_finder
from pathlib import Path
import pandas as pd


def run_measure(task_folder_path, model_name):
    # model architecture를 불러오고, 학습된 .pth 모델 가중치를 불러온다. 객체에 따라 학습된 모델 파일(.pth)이 달라질 것이다.
    model = pose_hrnet.get_pose_net(is_train=False).cuda()
    pretraind_model = f"api/measure/models/{model_name}.pth"
    model.load_state_dict(torch.load(pretraind_model), strict=True)

    c, s, r = np.array([719.5, 959.5]), np.array([7, 7]), 0  # center, scale, rotation
    # 이미지 크기에 맞는 [1920, 1440] c, s, r 이다.
    
    img_file = str(task_folder_path / 'image.jpg')
    
    model_folder_path = f"api/measure/bbox_models"
    model_file_path = os.path.join(model_folder_path, f'{model_name}.pt')
    if os.path.isfile(model_file_path):
        c, s = bbox_finder(task_folder_path,model_file_path, model_name, img_file)
        c, s, r = np.array(c), np.array(s), 0  # center, scale, rotation

    # 필요한 transform 불러오기
    trans = get_affine_transform(c, s, r, [288, 384])

    # evaluation mode
    model.eval()

    # [1920, 1440] 크기의 이미지 파일
    # img_file = str(task_folder_path / 'image.jpg')

    # input 전처리
    data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (288, 384), # 이 모델의 input size는 (288, 384)로 고정이다.
        flags=cv2.INTER_LINEAR)
    
    
    input_img_rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    # 6. 변환된 이미지 저장
    save_path = str(task_folder_path / 'transformed_image.jpg')
    cv2.imwrite(save_path, input_img_rgb)
    

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    it = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    input = it(input)
    input = torch.unsqueeze(input, 0).cuda()

    with torch.no_grad():

        cat_idx = 1 # gt_class_keypoints_dict 즉, 어떤 객체 클래스인지를 말한다. 여기서는 반팔로 테스트할 것이기 때문에 1을 넣었는데,
        # 아마 모든 클래스를 1로 학습했을 것이기 때문에 1을 그대로 사용하면 될 것이다. 물론 1: (n, n)은 바꿔야 함.

        channel_mask = torch.zeros((1, 294, 1)).cuda(non_blocking=True).float()
        cat_ids = [cat_idx]

        # gt_class_keypoints_dict를 통해 해당 클래스에서 찾을 점들의 인덱스 중 처음과 끝을 저장
        cat_idx_start = class_dict[model_name][int(cat_idx)][0]
        cat_idx_end = class_dict[model_name][int(cat_idx)][1]

        # 해당 클래스에서 찾을 점들을 인덱싱하기 위한 channel_mask 변경
        for j, cat_id in enumerate(cat_ids):
            rg = class_dict[model_name][int(cat_id)]
            index = torch.tensor([list(range(rg[0], rg[1]))], device=channel_mask.device,
                                 dtype=channel_mask.dtype).transpose(1, 0).long()
            channel_mask[j].scatter_(0, index, 1)

        output = model(input)
        output = output * channel_mask.unsqueeze(3)

        # preds_local은 작은 사이즈 (정확히는 히트맵 사이즈)에서 찾은 점들의 위치이다.
        preds_local, maxvals = get_final_preds(output.detach().cpu().numpy(), c, s)

        # preds로 원래 이미지 크기 [1920, 1400] 에서의 점의 위치를 찾는다.
        preds = preds_local.copy()
        for tt in range(preds_local.shape[0]):
            preds[tt] = transform_preds(
                preds_local[tt], c, s,
                [72, 96]
            )

        f_preds = preds[0][cat_idx_start:cat_idx_end]

    key_result = np.array(f_preds) # 최종 keypoints 좌표들

    # 예측한 keypoints 좌표들에 점 찍은 이미지 생성하고 확인 (필요없으면 생략)
    k = cv2.imread(img_file)
    for i in range(len(key_result)):
        cv2.circle(k, (int(key_result[i, 0]), int(key_result[i, 1])), 8, [255, 0, 0], -1)
        cv2.putText(k, str(i),(int(key_result[i, 0]), int(key_result[i, 1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0, color=(255, 20, 147), thickness=2, lineType=cv2.LINE_AA,)
    final_img_path = str(task_folder_path/'key_result.png')
    cv2.imwrite(final_img_path, k)
    
    
     # measure distance within detected keypoints
    pointcloud_path = str(task_folder_path/'depth.json')
    rgb_h = data_numpy.shape[0]

    world_xyz_array = get_world_xyz_array(pointcloud_path, rgb_h)


    rgb_depth_ratio = 7.5
    print("rgb_depth_ratio : ", rgb_depth_ratio)
    print("key_result : ", key_result)
    print("measure_dict is : ", measure_dict[model_name] )
    measure_result_dict, key_length_image = measure_distance(measure_dict[model_name], key_result, world_xyz_array, rgb_depth_ratio, k)
    measure_3d_result_dict = measure_3D_distance(measure_dict[model_name], key_result, world_xyz_array, rgb_depth_ratio)

    print("measure result", measure_result_dict, sep="\n")
    print("measure 3d result", measure_3d_result_dict, sep="\n")

    final_img_path = str(task_folder_path/'key_measure_result.png')
    cv2.imwrite(final_img_path, key_length_image)
    return  final_img_path
   



if __name__ == '__main__':
    run_measure()

