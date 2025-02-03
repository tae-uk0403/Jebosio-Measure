import numpy as np
import json
import cv2


# def get_world_xyz_array(pointcloud_path, rgb_h):
#     """
#     :param pointcloud_path: json 파일. cx, cy, 초기 z값이 쌍으로 있어야 한다. 또한, Camera Parameter 정보가 필요하다.
#     cx, cy, z : [cx : 0~depth_w, cy : 0~depth_h, z : initial value of z at taking a depth]
#     Intrinsic Parameters : [Focal Length : "fl x, y", Principal point : "depth width and height"]
#     Extrinsic Parameters : [Rotation Matrix : "rot", Translation Vector : "pos x, y, z"]
#     :param rgb_h: height of the rgb image
#     :return: world_xyz_array, shape : [depth_w, depth_h, 3]
#     """
#     with open (pointcloud_path, "r") as f:
#         depth_json = json.load(f)
#         # depth_w = depth_json['Width']
#         # depth_h = depth_json['Height']
#         depth_w = 256
#         depth_h = 192
#         fl_x = depth_json["fl"]["x"]
#         fl_y = depth_json["fl"]["y"]
#         rgb_depth_ratio = rgb_h/depth_h

#         camera_depth_array = np.zeros((depth_w, depth_h), dtype=np.float64)
#         for i in range(len(depth_json['Depth'])):
#             x=i%depth_w
#             y=i//depth_w
#             camera_depth_array[x][y] = depth_json['Depth'][i]


#         camera_xyz_array = []

#         # 거리를 실측할 수 있는 world_xyz_array를 만들기 전에 먼저 camera_xyz_array를 만든다.
#         # height부터 채우느냐, i부터 채우느냐에 따라 마지막 world_xyz_array가 달라질 수 있다.
#         # 현재 (24. 04. 18) 반팔 옷에 대해 테스트할 때는 height부터 채워야 실측이 제대로 된다.
#         for i in range(depth_w):
#             for j in range(depth_h):
#                 z = camera_depth_array[i][j]
#                 x = (j - depth_w / 2) * z / (fl_x / rgb_depth_ratio)
#                 y = (i - depth_h / 2) * z / (fl_y / rgb_depth_ratio)
#                 camera_xyz_array.append([x, y, z])

#         camera_xyz_array = np.array(camera_xyz_array)

#         # world_xyz_array를 만들기 위한 Rotation Matrix 만들기
#         R_camera_to_world = np.array(
#             [[depth_json['m00'], depth_json['m01'], depth_json['m02'], depth_json['m03']],
#              [depth_json['m10'], depth_json['m11'], depth_json['m12'], depth_json['m13']],
#              [depth_json['m20'], depth_json['m21'], depth_json['m22'], depth_json['m23']],
#              [depth_json['m30'], depth_json['m31'], depth_json['m32'], depth_json['m33']]])
#         R_camera_to_world = R_camera_to_world[0:3, 0:3]

#         # world_xyz_array를 만들기 위한 Translation Vector 만들기
#         T_camera_to_world = np.array([depth_json['Pos']['x'], depth_json['Pos']['y'], depth_json['Pos']['z']])

#         # camera_xyz_array, Rotation Matrix, Translation Vector를 활용하여 world_xyz_array를 만든다.
#         world_xyz_array = np.dot(camera_xyz_array, R_camera_to_world) + T_camera_to_world

#         # [256, 192, 3] shape으로 만든다.
#         world_xyz_array = world_xyz_array.reshape((256, 192, 3))
#         # world_xyz_array = np.flip(world_xyz_array, axis=1)

#         return world_xyz_array


def get_world_xyz_array(pointcloud_path, rgb_h):
    
    with open(pointcloud_path) as f:
        depth_json = json.load(f)
    depth_values = depth_json["Depth"]
    depth_w = 256
    depth_h = 192
    fl_x = depth_json['fl']['x']
    fl_y = depth_json['fl']['y']
    xyz_array = []
    k = 0
    for i in range(depth_h):
        for j in range(depth_w):
            z = depth_values[k]
            x = (j - 256 / 2) * z / (fl_x / 7.5)
            y = (i - 192 / 2) * z / (fl_y / 7.5)
            xyz_array.append([x, y, z])
            k += 1

    R_camera_to_world = np.array([[depth_json['m00'], depth_json['m01'], depth_json['m02'], depth_json['m03']],
                                  [depth_json['m10'], depth_json['m11'], depth_json['m12'], depth_json['m13']],
                                  [depth_json['m20'], depth_json['m21'], depth_json['m22'], depth_json['m23']],
                                  [depth_json['m30'], depth_json['m31'], depth_json['m32'], depth_json['m33']]])

    R_camera_to_world = R_camera_to_world[0:3, 0:3]

    T_camera_to_world  = np.array([depth_json['Pos']['x'], depth_json['Pos']['y'], depth_json['Pos']['z']])

    world_xyz = np.dot(xyz_array, R_camera_to_world) + T_camera_to_world

    world_xyz_array = world_xyz.reshape((192, 256, 3))
    world_xyz_array = np.rot90(world_xyz_array, k=-1)

    return world_xyz_array


def measure_distance(measure_dict, detected_keypoints, world_xyz_array, rgb_depth_ratio, RGB_image):
    """
    :param measure_dict: class_dict.py에 정의하는 class별 측정 대상에 대한 정보 dictionary. key : 측정 대상의 이름, value: 측정 대상이 되는 두 개의 keypoint index
    :param detected_keypoints: rgb image를 keypoint detection 모델에 입력하여 나온 결과. [x, y] 쌍으로 이루어진 점 좌표 리스트. shape : [점 개수, 2]
    :param world_xyz_array: [256, 192, 3] 형태의 world_xyz_array
    :param rgb_depth_ratio: height of rgb / height of depth or height of rgb / height of world_xyz_array 아마도 height of world_xzy_array = height of depth
    :return: 측정 대상별 길이 dictionary
    """
  
    measure_result_dict = {}

    for measure_name, measure_points in measure_dict.items():
        print('measure points is ' , measure_points)
        print('measure dict is' , measure_dict)
        
        target_point_1 = np.array(np.array(detected_keypoints[measure_points[0] - 1]) / rgb_depth_ratio, dtype=np.int64)
        target_point_2 = np.array(np.array(detected_keypoints[measure_points[1] - 1]) / rgb_depth_ratio, dtype=np.int64)
        
        print('first world_xyz_array is : ', world_xyz_array[target_point_1[1], target_point_1[0]])
        print('second world_xyz_array is : ', world_xyz_array[target_point_2[1], target_point_2[0]])
        
        measure_result_dict[measure_name] = np.linalg.norm(
            world_xyz_array[target_point_1[1], target_point_1[0]] - world_xyz_array[target_point_2[1], target_point_2[0]]) * 100
        
        detected_keypoints_1 = detected_keypoints[measure_points[0] - 1]
        detected_keypoints_2 = detected_keypoints[measure_points[1] - 1]
        print(f"measure name is {measure_name}, measure length is {measure_result_dict[measure_name]}, detected depth keypoints is {target_point_1}, {target_point_2}")
        
        cv2.line(RGB_image, (int(detected_keypoints_1[0]), int(detected_keypoints_1[1])), (int(detected_keypoints_2[0]), int(detected_keypoints_2[1])), (0, 255, 0), thickness=6, lineType=cv2.LINE_AA)
        cv2.putText(RGB_image, '{0} : {1:.2f}cm'.format(measure_name, measure_result_dict[measure_name]),(int((detected_keypoints_1[0] + detected_keypoints_2[0]) / 2), int((detected_keypoints_1[1] + detected_keypoints_2[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4)    
        
  

    return measure_result_dict, RGB_image


def abcd(a,b,c,d):
    """
    :return: 측정대상별 Dictionary = {"어깨":{"길이":54, "키포인트":[[1000, 500], [1100, 500]]}, "가슴":{}, ...} 
    """


def measure_3D_distance(measure_dict, detected_keypoints, world_xyz_array, rgb_depth_ratio):
    """
    :param measure_dict: class_dict.py에 정의하는 class별 측정 대상에 대한 정보 dictionary. key : 측정 대상의 이름, value: 측정 대상이 되는 두 개의 keypoint index
    :param detected_keypoints: rgb image를 keypoint detection 모델에 입력하여 나온 결과. [x, y] 쌍으로 이루어진 점 좌표 리스트. shape : [점 개수, 2]
    :param world_xyz_array: [256, 192, 3] 형태의 world_xyz_array
    :param rgb_depth_ratio: height of rgb / height of depth or height of rgb / height of world_xyz_array 아마도 height of world_xzy_array = height of depth
    :return: 측정 대상별 길이 dictionary
    """

    measure_3d_result_dict = {}

    for measure_name, measure_points in measure_dict.items():
        print('measure points is ' , measure_points)
        print('measure dict is' , measure_dict)

        target_point_1 = np.array(np.array(detected_keypoints[0]) / rgb_depth_ratio, dtype=np.int64)
        target_point_2 = np.array(np.array(detected_keypoints[1]) / rgb_depth_ratio, dtype=np.int64)
        
        length_3d = 0
        for i in range(target_point_1[0]-10,target_point_2[0]+10):    
            length_3d += np.linalg.norm(
                world_xyz_array[target_point_1[1],i] - world_xyz_array[target_point_1[1],i+1]) * 100
            # print(i, np.linalg.norm(
            #     world_xyz_array[target_point_1[1],i] - world_xyz_array[target_point_1[1],i+1]) * 100)
        measure_3d_result_dict[measure_name] = length_3d
    return measure_3d_result_dict


