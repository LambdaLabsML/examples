import datetime
import hashlib
import numpy as np
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2
import gradio as gr
from joblib import Parallel, delayed
from numpy.typing import NDArray
from PIL import Image


def _run_in_subprocess(command: str, wd: str) -> Any:
    p = subprocess.Popen(command, shell=True, cwd=wd)
    (output, err) = p.communicate()
    p_status = p.wait()
    print("Status of subprocess: ", p_status)
    return p_status


SWIN_IR_WD = "KAIR"
SWINIR_CKPT_DIR: str = Path("KAIR/model_zoo/")
MODEL_NAME_TO_PATH: Dict[str, Path] = {
    "LambdaSwinIR_v0.1": Path(str(SWINIR_CKPT_DIR) + "/805000_G.pth"),
}
SWINIR_NAME_TO_PATCH_SIZE: Dict[str, int] = {
    "LambdaSwinIR_v0.1": 96,
}
SWINIR_NAME_TO_SCALE: Dict[str, int] = {
    "LambdaSwinIR_v0.1": 2,
}
SWINIR_NAME_TO_LARGE_MODEL: Dict[str, bool] = {
    "LambdaSwinIR_v0.1": False,
}

def _run_swin_ir(
    image: NDArray,
    model_path: Path,
    patch_size: int,
    scale: int,
    is_large_model: bool,
):
    print("model_path: ", str(model_path))
    m = hashlib.sha256()
    now_time = datetime.datetime.utcnow()
    m.update(bytes(str(model_path), encoding='utf-8') +
             bytes(now_time.strftime("%Y-%m-%d %H:%M:%S.%f"), encoding='utf-8'))
    random_id = m.hexdigest()[0:20]

    cwd = os.getcwd()

    input_root = Path(cwd + "/sr_interactive_tmp")
    input_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(str(input_root) + "/gradio_img.png")
    command = f"python main_test_swinir.py --scale {scale} " + \
        f"--folder_lq {input_root} --task real_sr " + \
        f"--model_path {cwd}/{model_path} --training_patch_size {patch_size}"
    if is_large_model:
        command += " --large_model"
    print("COMMAND: ", command)
    status = _run_in_subprocess(command, wd=cwd + "/" + SWIN_IR_WD)
    print("STATUS: ", status)

    if scale == 2:
        str_scale = "2"
    if scale == 4:
        str_scale = "4_large"
    output_img = Image.open(f"{cwd}/KAIR/results/swinir_real_sr_x{str_scale}/gradio_img_SwinIR.png")
    output_root = Path("./sr_interactive_tmp_output")
    output_root.mkdir(parents=True, exist_ok=True)

    output_img.save(str(output_root) + "/SwinIR_" + random_id + ".png")
    print("SAVING: SwinIR_" + random_id + ".png")
    result = np.array(output_img)
    return result


def _bilinear_upsample(image: NDArray):
    result = cv2.resize(
        image,
        dsize=(image.shape[1] * 2, image.shape[0] * 2),
        interpolation=cv2.INTER_LANCZOS4
    )
    return result


def _decide_sr_algo(model_name: str, image: NDArray):
    # if "SwinIR" in model_name:
    #     result = _run_swin_ir(image,
    #                           model_path=MODEL_NAME_TO_PATH[model_name],
    #                           patch_size=SWINIR_NAME_TO_PATCH_SIZE[model_name],
    #                           scale=SWINIR_NAME_TO_SCALE[model_name],
    #                           is_large_model=("SwinIR-L" in model_name))
    # else:
    #     result = _bilinear_upsample(image)

    # elif algo == SR_OPTIONS[1]:
    #     result = _run_maxine(image, mode="SR")
    # elif algo == SR_OPTIONS[2]:
    #     result = _run_maxine(image, mode="UPSCALE")
    # return result
    result = _run_swin_ir(image,
                          model_path=MODEL_NAME_TO_PATH[model_name],
                          patch_size=SWINIR_NAME_TO_PATCH_SIZE[model_name],
                          scale=SWINIR_NAME_TO_SCALE[model_name],
                          is_large_model=SWINIR_NAME_TO_LARGE_MODEL[model_name])
    return result


def _super_resolve(model_name: str, input_img):
    # futures = []
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     for model_name in model_names:
    #         futures.append(executor.submit(_decide_sr_algo, model_name, input_img))

    # return [f.result() for f in futures]
    # return Parallel(n_jobs=2, prefer="threads")(
    #     delayed(_decide_sr_algo)(model_name, input_img)
    #     for model_name in model_names
    # )
    return _decide_sr_algo(model_name, input_img)

def _gradio_handler(sr_option: str, input_img: NDArray):
    return _super_resolve(sr_option, input_img)


gr.close_all()
SR_OPTIONS = ["LambdaSwinIR_v0.1"]
examples = [
    ["LambdaSwinIR_v0.1", "examples/oldphoto6.png"],
    ["LambdaSwinIR_v0.1", "examples/Lincoln.png"],
    ["LambdaSwinIR_v0.1", "examples/OST_009.png"],
    ["LambdaSwinIR_v0.1", "examples/00003.png"],
    ["LambdaSwinIR_v0.1", "examples/00000067_cropped.png"],
]
ui = gr.Interface(fn=_gradio_handler,
                  inputs=[
                      gr.Radio(SR_OPTIONS),
                      gr.Image(image_mode="RGB")
                  ],
                  outputs=["image"],
                  live=False,
                  examples=examples,
                  cache_examples=True)
ui.launch(enable_queue=True, share=True)
