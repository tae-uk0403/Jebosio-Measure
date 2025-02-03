import os
import os.path as osp

import sys
import os

# 프로젝트 루트 디렉토리 설정
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)




import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
from network.network import rgbdp_v2_twostep_model
from utils.functions import *
import cv2
from torchvision import transforms as T
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

def run_fas(task_folder_path):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cloudnet_v2 = rgbdp_v2_twostep_model(device=device)
    model_path = Path('api/fas') / "model/epoch_232_model.pth"
    cloud_v2 = torch.load(model_path)
    cloudnet_v2.load_state_dict(cloud_v2['model_state_dict'])

    sigmoid = nn.Sigmoid()
    loss_fn = nn.BCEWithLogitsLoss()    

    cloudnet_v2.eval()

    # input
    rgb_path = str(task_folder_path / 'image.jpg' )
    depth_path = str(Path('api/fas') / "data/sample_test/01_depth.jpg")
    pc_path = Path('api/fas') / "data/sample_test/01_pc.ply"

    print('rgb_path is ', rgb_path)
    print(depth_path)

    rgb = preprocess_data(rgb_path, "rgb").float().unsqueeze(0).to(device)
    depth = preprocess_data(depth_path, "depth").float().unsqueeze(0).to(device)
    pc = preprocess_data(pc_path, "pc").float().unsqueeze(0).to(device)

    logits_cloud_v2 = cloudnet_v2(rgb, depth, pc)[:, 0]
    prob_cloud_v2 = sigmoid(logits_cloud_v2).cpu().detach().tolist()[0]

    result = np.round(prob_cloud_v2) # 0 : bonafide, 1 : Spoofing
    

    if result == 0:
        answer = "Bonafide!"
    else :
        answer = "Spoofing!"
        
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (50, 50)
    font_scale = 1
    font_color = (255, 255, 255)  # 흰색
    line_type = 2

    # 이미지에 텍스트 추가
    final_img_path = str(task_folder_path / 'anti_spoofing_result_image.png')
    rgb_image = cv2.imread(rgb_path)
    text_image = cv2.putText(rgb_image, answer, bottom_left_corner_of_text, font, font_scale, font_color, line_type)
    cv2.imwrite(final_img_path, text_image)
    
    return answer
