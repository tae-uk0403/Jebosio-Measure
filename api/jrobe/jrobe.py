from mstr.jrobe.perform_test.utils import Jrobe


def run_jrobe(task_folder_path):
    img_pth = str(task_folder_path / "image.png")
    json_pth = str(task_folder_path / "depth.json")
    result_pth = str(task_folder_path / "output.png")
    
    J = Jrobe()
    J.pred_length(img_pth, json_pth, save=result_pth)
    return result_pth
