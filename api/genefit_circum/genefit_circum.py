from pathlib import Path
from mstr.genefit.perform_test.utils import GenefitCircumference


def run_genefit_circum(task_folder_path):
    img_pth = str(task_folder_path / "image.png")
    json_pth = str(task_folder_path / "depth.json")
    result_pth = str(task_folder_path / "output.png")
    
    CM = GenefitCircumference()
    CM.pred_ccfc(img_pth, json_pth, save=result_pth)
    return result_pth