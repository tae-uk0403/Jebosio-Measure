from pathlib import Path
from mstr.genefit.perform_test.utils import GNF_LA


def run_genefit(task_folder_path, pose):

    LAM = GNF_LA()
    img_pth = str(task_folder_path / "image.png")
    json_pth = str(task_folder_path / "depth.json")
    result_pth = str(task_folder_path / "output.png")

    LAM.pred_pose_len_ag(img_pth, json_pth, pose, agopt="2d", save=result_pth)
    return result_pth
