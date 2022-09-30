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
    "SwinIR-L_x4": Path(str(SWINIR_CKPT_DIR) + "/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"),
}
SWINIR_NAME_TO_PATCH_SIZE: Dict[str, int] = {
    "LambdaSwinIR_v0.1": 96,
    "SwinIR-L_x4": 64,
}
SWINIR_NAME_TO_SCALE: Dict[str, int] = {
    "LambdaSwinIR_v0.1": 2,
    "SwinIR-L_x4": 4,
}
SWINIR_NAME_TO_LARGE_MODEL: Dict[str, bool] = {
    "LambdaSwinIR_v0.1": False,
    "SwinIR-L_x4": True,
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


def _lanczos_upsample(image: NDArray):
    result = cv2.resize(
        image,
        dsize=(image.shape[1] * 2, image.shape[0] * 2),
        interpolation=cv2.INTER_LINEAR
    )
    return result

SIZE_LIMIT = 1024

def _downsize_if_necessary(image: NDArray):
    h, w, c = image.shape
    new_h = h
    new_w = w
    if h > SIZE_LIMIT:
        new_h = SIZE_LIMIT
    if w > SIZE_LIMIT:
        new_w = SIZE_LIMIT

    if h == new_h and w == new_w:
        return image

    result = cv2.resize(
        image,
        dsize=(new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )
    return result



def _decide_sr_algo(model_name: str, image: NDArray):
    image = _downsize_if_necessary(image)
    if model_name == "Naive upsample":
        return _lanczos_upsample(image)
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
    return _decide_sr_algo(model_name, input_img)


def _gradio_handler(sr_option: str, input_img: NDArray):
    if sr_option not in SR_OPTIONS:
        sr_option = "LambdaSwinIR_v0.1"
    return _super_resolve(sr_option, input_img)


def gradio_auth(username: str, password: str) -> bool:
    if username == "deepvoodoo":
        if password == "super_resolution":
            return True
    return False


def _clear_on_click(input_img, output_img):
    input_img = None
    output_img = None


def _clear_viewer(viewer_image: NDArray):
    viewer_image = None


def _save_on_click(name: str, output_img: NDArray):
    Image.fromarray(output_img).save(os.path.expanduser("~") + "/Desktop/" + name + ".png")


gr.close_all()
SR_OPTIONS = ["LambdaSwinIR_v0.1", "SwinIR-L_x4", "Naive upsample"]
EXAMPLES_DIR = Path("examples")

example_files = EXAMPLES_DIR.glob('**/*')
examples_sorted = []
for example_file in example_files:
    examples_sorted.append(str(Path(example_file)))
examples_sorted.sort()

examples = []
for option in SR_OPTIONS:
    for example in examples_sorted:
        examples.append([option, example])

global_sr_option: str = None

with gr.Blocks() as ui:
    with gr.Tab("Super Res"):
        with gr.Row():
            with gr.Column():
                sr_option = gr.Radio(SR_OPTIONS, value="LambdaSwinIR_v0.1",
                                     label="Select super res algo", interactive=True),
                global_sr_option = sr_option[0].cleared_value

                warning = gr.Markdown(value=f"##### NOTE: images larger than {SIZE_LIMIT}x{SIZE_LIMIT} will be downsampled.")
                input_img = gr.Image(
                    image_mode="RGB",
                    label="Input image")

            with gr.Column():
                output_img = gr.Image(image_mode="RGB", label="Output", interactive=False)

        with gr.Row():
            submit_button = gr.Button("Submit")
            submit_button.click(_gradio_handler,
                                inputs=[sr_option[0], input_img],
                                outputs=[output_img])

    # TODO: This could be useful later.
    # with gr.Tab("Viewer"):
    #     viewer_images = gr.Files(label="Images to view", interactive=True)
    #     # viewer = gr.Gallery(value=viewer_images, label="Gallery")
    examples = gr.Examples(examples=examples, inputs=[sr_option[0], input_img],
                           outputs=output_img, fn=_gradio_handler, cache_examples=True)

ui.launch(share=True)  # , auth=gradio_auth)
