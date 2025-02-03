from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np



def fix_bbox_xylist(width, height, x, y, w, h):
    valid_objs = []
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    print(x1, x2, y1, y2)
    return _xywh2cs(width, height, x1, y1, x2-x1, y2-y1)


def _xywh2cs(width, height, x, y, w, h):
    aspect_ratio = width / height
    
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / 200, h * 1.0 / 200],
        dtype=np.float32)
    # if center[0] != -1:
        # scale = scale * 1.25
    
    return center, scale




def bbox_finder(task_folder_path,model_file_path, model_name, image_path):
    # 미리 학습된 YOLO 모델을 불러옵니다.
    model = YOLO(model_file_path)  # 학습 완료된 모델 파일을 지정합니다.

    # 이미지를 로드하고 바운딩 박스를 감지합니다.
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    width, height = img.size


    # 결과로부터 바운딩 박스 좌표와 신뢰도 점수를 얻습니다.
    results = model(image_path)
    print('result is :')
    # 가장 높은 신뢰도를 가진 바운딩 박스를 저장할 변수를 초기화합니다.
    best_box = None
    best_conf = -1  # 신뢰도 점수는 0에서 1 사이이므로 초기값을 -1로 설정

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            # 현재 바운딩 박스의 신뢰도 점수가 이전에 기록된 것보다 높은지 확인합니다.
            if conf > best_conf:
                best_conf = conf
                best_box = (box, cls)
                print("best_box = ", best_box)

    # 신뢰도가 가장 높은 바운딩 박스가 있다면 그립니다.
    if best_box is not None:
        box, cls = best_box
        x1, y1, x2, y2 = map(int, box)
        # 바운딩 박스를 이미지에 그립니다.
        draw.rectangle([x1, y1, x2, y2], outline='green', width=3)

        # 클래스 이름을 바운딩 박스 위에 표시합니다.
        class_name = model.names[int(cls)]
        draw.text((x1, y1 - 10), class_name, fill='green')
        
        img.save(task_folder_path / 'detection_box.png')

        
        print("original is : ", x1, y1, x2, y2)
        w = max(x1, x2) - min(x1, x2)
        h = max(y1, y2) - min(y1, y2)
        
    try :    
        answer = fix_bbox_xylist(width, height, min(x1, x2), min(y1, y2), w, h)
    except : 
        fix_bbox_xylist(width, height, min(x1, x2), min(y1, y2), w, h)
    return fix_bbox_xylist(width, height, min(x1, x2), min(y1, y2), w, h)