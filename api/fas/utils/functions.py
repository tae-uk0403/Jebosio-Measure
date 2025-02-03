from torchvision import transforms as T
import cv2
import numpy as np
from plyfile import PlyData
import torch


def preprocess_data(data_path, category=""):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms = T.Compose([
        T.ToTensor(),
        normalize
    ])

    transforms2 = T.Compose([
        T.ToTensor()
    ])

    crop_width = 90
    crop_height = 150
    mid_x, mid_y = 90, 90
    offset_x, offset_y = crop_width // 2, crop_height // 2

    if category == "rgb":
        rgb_data = cv2.imread(data_path)
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = cv2.resize(rgb_data, (180, 180), interpolation=cv2.INTER_CUBIC)
        # rgb_data = rgb_data[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x]
        rgb_data = transforms(rgb_data)

        return rgb_data

    if category == "depth":
        depth_data = cv2.imread(data_path)
        depth_data = cv2.cvtColor(depth_data, cv2.COLOR_BGR2GRAY)
        depth_data = cv2.resize(depth_data, (180, 180), interpolation=cv2.INTER_CUBIC)
        depth_data = transforms2(depth_data)

        return depth_data

    if category == "pc":
        cloud_data = ply_to_npy(data_path)
        cloud_data = cv2.resize(cloud_data, (180, 180), interpolation=cv2.INTER_CUBIC)
        cloud_data += 5
        # cloud_data = cloud_data[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x]

        # Point Cloud and Depth Scaling
        shift_value = 0
        xcoor = np.array(cloud_data[:, :, 0] + shift_value)
        ycoor = np.array(cloud_data[:, :, 1] + shift_value)
        zcoor = np.array(cloud_data[:, :, 2] + shift_value)
        xcoor = (xcoor - xcoor.min()) / (xcoor.max() - xcoor.min())
        ycoor = (ycoor - ycoor.min()) / (ycoor.max() - ycoor.min())
        zcoor = (zcoor - zcoor.min()) / (zcoor.max() - zcoor.min())
        scaled_cloud_data = np.concatenate([xcoor[np.newaxis, :], ycoor[np.newaxis, :], zcoor[np.newaxis, :]])
        scaled_cloud_data = torch.Tensor(scaled_cloud_data)

        return scaled_cloud_data


def ply_to_npy(ply_path):
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(ply_path)

    num_verts = plydata['vertex'].count
    vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
    vertices[:, 0] = plydata['vertex'].data['x']
    vertices[:, 1] = plydata['vertex'].data['y']
    vertices[:, 2] = plydata['vertex'].data['z']
    vertices[:, 3] = plydata['vertex'].data['cx']
    vertices[:, 4] = plydata['vertex'].data['cy']
    vertices[:, 5] = plydata['vertex'].data['depth']

    proj_xyzd = np.full((4, 192, 256), -1,dtype=np.float32)

    for k in range(len(vertices)):
        proj_xyzd[0, int(vertices[k, 4]), int(vertices[k, 3])] = vertices[k, 0]
        proj_xyzd[1, int(vertices[k, 4]), int(vertices[k, 3])] = vertices[k, 1]
        proj_xyzd[2, int(vertices[k, 4]), int(vertices[k, 3])] = vertices[k, 2]
        proj_xyzd[3, int(vertices[k, 4]), int(vertices[k, 3])] = vertices[k, 5]

    proj_xyzd = proj_xyzd.transpose(1, 2, 0)

    return proj_xyzd


