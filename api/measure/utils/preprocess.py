import cv2
import matplotlib.pyplot as plt


def depth_to_xyz(depth_image_path):
    depth_img = cv2.imread("data/cylinder/sample/2024-04-16 PM 12_23_11_depth.png", cv2.IMREAD_GRAYSCALE)
    depth_img