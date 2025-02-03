from datetime import datetime
from pathlib import Path
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Query,
    Body,
    status,
    Request,
    Response,
    Depends,

    HTTPException,
    Header,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware


from typing import Annotated
from plyfile import PlyData
import time
import logging
import os
from api_key.verify_api_key import verify_api_key

# from inpainting.inpainting import run_in_painting_with_stable_diffusion
from api.img_kpt.img_kpt import run_img_kpt_processing
from api.measure.measure import run_measure
from api.fas.fas import run_fas
from api.jrobe_circum.jrobe_circum import run_jrobe_circum
from api.jrobe.jrobe import run_jrobe
from api.genefit.genefit import run_genefit
from api.genefit_circum.genefit_circum import run_genefit_circum


import zipfile
import json
import shutil


# Load API description from a markdown file
with open(Path("DESCRIPTION.md"), "r") as f:
    api_desc = f.read()

with open("error_code.json", "r") as f:
    responses = json.load(f)


tags_metadata = [
    {
        "name": "API Reference",
    },
]


app = FastAPI(
    title="Jebosio-Meausre API",
    description=api_desc,
    version="0.0.0",
    openapi_tags=tags_metadata,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


logging.basicConfig(filename="app.log", level=logging.INFO)


# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 요청 헤더 허용
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    api_key = request.headers.get("api-key")

    print("api_key is ", api_key)
    if api_key is not None:

        log_dir = "api_key/api_info.json"
        with open(log_dir, "r", encoding="utf-8") as file:
            data = json.load(file)

        for company_info in data["info"]:
            if company_info.get("key") == api_key:
                Company = company_info.get("company", None)

        # 요청 정보 기록
        log_entry = {
            "user": Company,
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": process_time,
        }

        logging.info(log_entry)

    # 요청 정보 반환
    return response


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    pass


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Hello API"}


@app.get("/logs", include_in_schema=False)
async def get_logs():
    with open("app.log", "r") as file:
        logs = file.readlines()
    return logs


@app.post(
    "/api/v1.0/cloth/img-kpt",
    tags=["API Reference"],
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    responses={
        **responses,
        200: {
            "content": {"image/png": {"example": "(binary image data)"}},
            "description": "Created",
        },
    },
)
async def image_keypoint(
    image_file: Annotated[
        UploadFile, File(media_type="image/png", description="Image file")
    ],
    depth_file: Annotated[
        UploadFile, File(media_type="image/png", description="Depth image")
    ],
    api_key: str = Depends(verify_api_key)
):
    """
    - **Description**: This endpoint performs image processing and keypoint extraction. Clients upload image files and other necessary parameters for the operation.

    - **Request Parameters**:
        - **`image_file`**: Image for the size measurement (upload)
            - **Example**:

                <img src='http://203.252.147.202:8000/static/img-kpt/image_file.jpg' width=350 height=300>
        - **`depth_file`**: Image depth image (upload)


    - **Response**: Processed image file (PNG format)
        - **Example**:

            ![https://i.imgur.com/OnO1yhum.png](https://i.imgur.com/OnO1yhum.png)
    """

    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = (
        Path("api/img_kpt/temp_process_task_files") / api_key / sub_folder_name
    )
    task_id = sub_folder_name

    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path("image_file.jpg")
    depth_file_path = task_folder_path / Path("depth_file.jpg")

    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    with depth_file_path.open("wb") as f:
        f.write(await depth_file.read())

    run_img_kpt_processing(
        task_folder_path=task_folder_path, clothes_type=1, model_version=2
    )

    # Return the result file as a response
    result_image_path = task_folder_path / "result_image_v1.png"
    return FileResponse(
        result_image_path,
        status_code=status.HTTP_200_OK,
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_img_kpt_image_v1.png"
        },
    )


@app.post(
    "/api/v1.0/fas",
    tags=["API Reference"],
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    responses={
        **responses,
        200: {
            "content": {"text/plain": {"example": "(text data)"}},
            "description": "Created",
        },
    },
)
async def fas(
    # json_file: Annotated[UploadFile, File(media_type="text/json",
    #                                      description="apple ply for size measurement")],
    image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="Face image to check the reality"),
    ],
    api_key: str = Depends(verify_api_key),
):
    """
    - **Description**: This endpoint measures the size of a strawberry from the provided image.

    - **Request Parameters**:
        - **`image_file`**: Face image to check the reality (upload)
            - **Example**:

                <img src='http://203.252.147.202:8000/static/fas/image.jpg' width=200 height=200>

    - **Response**: Image file with the face and text whether the image is fake or not (PNG format)
        - **Example**:

            <img src='http://203.252.147.202:8000/static/fas/anti_spoofing_result_image.png' width=200 height=200>
    """
    if not image_file:
        return {"error": "No file uploaded"}

    print("request succeed")
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = (
        Path("api/fas") / "temp_process_task_files" / api_key / sub_folder_name
    )
    print("task_folder_path is ", task_folder_path)
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path("image.jpg")
    image_file_path.parent.mkdir(parents=True, exist_ok=True)
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    result_image_path = run_fas(task_folder_path)
    print("result_image_path : ", result_image_path)
    return PlainTextResponse(result_image_path)


@app.post(
    "/api/v1.0/jrobe",
    tags=["API Reference"],
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    responses={
        **responses,
        200: {
            "content": {"image/png": {"example": "(binary image data)"}},
            "description": "Created",
        },
    },
)
async def jrobe(
    front_image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="image for size measurement"),
    ],
    front_json_file: Annotated[
        UploadFile,
        File(media_type="text/json", description="apple ply for size measurement"),
    ],
    side_image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="image for size measurement"),
    ],
    side_json_file: Annotated[
        UploadFile,
        File(media_type="text/json", description="apple ply for size measurement"),
    ],
    api_key: str = Depends(verify_api_key),
):
    """
    - **Description**: This endpoint measures the size of a strawberry from the provided image.

    - **Request Parameters**:
        - **`image_file`**: Strawberry image for size measurement (upload)
            - **Example**:

                ![https://i.imgur.com/gAQt0lLm.jpg](https://i.imgur.com/gAQt0lLm.jpg)
    - **Response**: Image file with the strawberry's size measurement (PNG format)
        - **Example**:

            ![https://i.imgur.com/DzQazmGm.jpg](https://i.imgur.com/DzQazmGm.jpg)
    """
    if not front_image_file:
        return {"error": "No file uploaded"}
    print("request succeed")
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path("api/jrobe") / \
        "temp_process_task_files" / sub_folder_name
    print("task_folder_path is ", task_folder_path)
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    front_image_file_path = task_folder_path / Path("image.png")
    front_image_file_path.parent.mkdir(parents=True, exist_ok=True)
    with front_image_file_path.open("wb") as f:
        f.write(await front_image_file.read())

    front_json_file_path = task_folder_path / Path("depth.json")
    front_json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with front_json_file_path.open("wb") as f:
        f.write(await front_json_file.read())

    result_image_path = run_jrobe(task_folder_path)

    return FileResponse(
        result_image_path,
        media_type="image/png",
        status_code=status.HTTP_200_OK,
        headers={"Content-Disposition": f"attachment; filename={task_id}_image.png"},
    )


@app.post(
    "/api/v1.0/jrobe_circum",
    tags=["API Reference"],
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    responses={
        **responses,
        200: {
            "content": {"image/png": {"example": "(binary image data)"}},
            "description": "Created",
        },
    },
)
async def jrobe_circum(
    front_image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="image for size measurement"),
    ],
    front_json_file: Annotated[
        UploadFile,
        File(media_type="text/json", description="apple ply for size measurement"),
    ],
    side_image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="image for size measurement"),
    ],
    side_json_file: Annotated[
        UploadFile,
        File(media_type="text/json", description="apple ply for size measurement"),
    ],
    api_key: str = Depends(verify_api_key),
):
    """
    - **Description**: This endpoint measures the size of a strawberry from the provided image.

    - **Request Parameters**:
        - **`image_file`**: Strawberry image for size measurement (upload)
            - **Example**:

                ![https://i.imgur.com/gAQt0lLm.jpg](https://i.imgur.com/gAQt0lLm.jpg)
    - **Response**: Image file with the strawberry's size measurement (PNG format)
        - **Example**:

            ![https://i.imgur.com/DzQazmGm.jpg](https://i.imgur.com/DzQazmGm.jpg)
    """
    if not front_image_file:
        return {"error": "No file uploaded"}
    print("request succeed")
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path("api/jrobe_circum") / \
        "temp_process_task_files" / sub_folder_name
    print("task_folder_path is ", task_folder_path)
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    front_image_file_path = task_folder_path / Path("image.png")
    front_image_file_path.parent.mkdir(parents=True, exist_ok=True)
    with front_image_file_path.open("wb") as f:
        f.write(await front_image_file.read())

    front_json_file_path = task_folder_path / Path("depth.json")
    front_json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with front_json_file_path.open("wb") as f:
        f.write(await front_json_file.read())

    result_image_path = run_jrobe_circum(task_folder_path)

    return FileResponse(
        result_image_path,
        media_type="image/png",
        status_code=status.HTTP_200_OK,
        headers={"Content-Disposition": f"attachment; filename={task_id}_image.png"},
    )


@app.post(
    "/api/v1.0/genefit/{pose}",
    tags=["API Reference"],
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    responses={
        **responses,
        200: {
            "content": {"image/png": {"example": "(binary image data)"}},
            "description": "Created",
        },
    },
)
async def genefit(
    image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="image for size measurement"),
    ],
    json_file: Annotated[
        UploadFile,
        File(media_type="text/json", description="apple ply for size measurement"),
    ],
    pose: str,
    api_key: str = Depends(verify_api_key),
):
    """
    - **Description**: This endpoint measures the angle of work out pose

    - **Request Parameters**:
        - **`image_file`**: Strawberry image for size measurement (upload)
            - **Example**:

                ![https://i.imgur.com/gAQt0lLm.jpg](https://i.imgur.com/gAQt0lLm.jpg)
    - **Response**: Image file with the strawberry's size measurement (PNG format)
        - **Example**:

            ![https://i.imgur.com/DzQazmGm.jpg](https://i.imgur.com/DzQazmGm.jpg)
    """
    if not image_file:
        return {"error": "No file uploaded"}
    print("request succeed")
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path("api/genefit") / \
        "temp_process_task_files" / sub_folder_name
    print("task_folder_path is ", task_folder_path)
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path("image.png")
    image_file_path.parent.mkdir(parents=True, exist_ok=True)
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    json_file_path = task_folder_path / Path("depth.json")
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with json_file_path.open("wb") as f:
        f.write(await json_file.read())

    result_image_path = run_genefit(task_folder_path, pose)

    return FileResponse(
        result_image_path,
        media_type="image/png",
        status_code=status.HTTP_200_OK,
        headers={"Content-Disposition": f"attachment; filename={task_id}_image.png"},
    )


@app.post(
    "/api/v1.0/genefit_circum",
    tags=["API Reference"],
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    responses={
        **responses,
        200: {
            "content": {"image/png": {"example": "(binary image data)"}},
            "description": "Created",
        },
    },
)
async def genefit_circum(
    image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="image for size measurement"),
    ],
    json_file: Annotated[
        UploadFile,
        File(media_type="text/json", description="apple ply for size measurement"),
    ],
    api_key: str = Depends(verify_api_key),
):
    """
    - **Description**: This endpoint measures the angle of work out pose

    - **Request Parameters**:
        - **`image_file`**: Strawberry image for size measurement (upload)
            - **Example**:

                ![https://i.imgur.com/gAQt0lLm.jpg](https://i.imgur.com/gAQt0lLm.jpg)
    - **Response**: Image file with the strawberry's size measurement (PNG format)
        - **Example**:

            ![https://i.imgur.com/DzQazmGm.jpg](https://i.imgur.com/DzQazmGm.jpg)
    """
    if not image_file:
        return {"error": "No file uploaded"}
    print("request succeed")
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path("api/genefit_circum") / \
        "temp_process_task_files" / sub_folder_name
    print("task_folder_path is ", task_folder_path)
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path("image.png")
    image_file_path.parent.mkdir(parents=True, exist_ok=True)
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    json_file_path = task_folder_path / Path("depth.json")
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with json_file_path.open("wb") as f:
        f.write(await json_file.read())

    result_image_path = run_genefit_circum(task_folder_path)

    return FileResponse(
        result_image_path,
        media_type="image/png",
        status_code=status.HTTP_200_OK,
        headers={"Content-Disposition": f"attachment; filename={task_id}_image.png"},
    )


@app.post(
    "/api/v1.0/measure/{model_name}",
    tags=["API Reference"],
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    responses={
        **responses,
        200: {
            "content": {"image/png": {"example": "(binary image data)"}},
            "description": "Created",
        },
    },
)
async def measure_size(
    image_file: Annotated[
        UploadFile,
        File(media_type="image/png", description="image for size measurement"),
    ],
    model_name: str,
    json_file: Annotated[
        UploadFile,
        File(media_type="text/json", description="apple ply for size measurement"),
    ],
):
    """
    - **Description**: This endpoint measures the size of a strawberry from the provided image.

    - **Request Parameters**:
        - **`image_file`**: Strawberry image for size measurement (upload)
            - **Example**:

                ![https://i.imgur.com/gAQt0lLm.jpg](https://i.imgur.com/gAQt0lLm.jpg)
    - **Response**: Image file with the strawberry's size measurement (PNG format)
        - **Example**:

            ![https://i.imgur.com/DzQazmGm.jpg](https://i.imgur.com/DzQazmGm.jpg)
    """
    if not image_file:
        return {"error": "No file uploaded"}
    print("request succeed")
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = (
        Path("api/measure") / "temp_process_task_files" /
        model_name / sub_folder_name
    )
    print("task_folder_path is ", task_folder_path)
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path("image.jpg")
    image_file_path.parent.mkdir(parents=True, exist_ok=True)
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    json_file_path = task_folder_path / Path("depth.json")
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with json_file_path.open("wb") as f:
        f.write(await json_file.read())

    if model_name.startswith("cloth"):
        cloth_type = model_name[5:]
        model_name = model_name[0:6]

    result_image_path = run_measure(task_folder_path, model_name)

    return FileResponse(
        result_image_path,
        media_type="image/png",
        status_code=status.HTTP_200_OK,
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_{model_name}_image.png"
        },
    )
