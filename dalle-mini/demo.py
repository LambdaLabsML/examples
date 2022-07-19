import streamlit as st
import torch
from torchvision import utils
from PIL import Image

from min_dalle import MinDalle


st.title("dalle-mini (mega) + Real-ESRGAN")

@st.cache(allow_output_mutation=True)
def load_model():
    model = MinDalle(
        models_root='./pretrained',
        dtype=torch.float32,
        device='cuda:0',
        is_mega=True,
        is_reusable=True
    )
    return model

@st.cache(allow_output_mutation=True)
def load_upscaler():
    from basicsr.archs.rrdbnet_arch import RRDBNet

    import sys
    sys.path.append("./Real-ESRGAN")
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    model_path = "./RealESRGAN_x4plus.pth"

    # restorer
    upscaler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        gpu_id=0)
    return upscaler

@torch.no_grad()
def run(text, model):
    image = model.generate_images(
        text=text,
        seed=0,
        grid_size=2,
    )
    return image

@torch.no_grad()
def upscale(image, upscaler, factor=2):
    if factor < 4:
        image = torch.nn.functional.interpolate(image, int(64*factor), mode='area')
    return upscaler.model(image)

text = st.text_input("prompt")
go = st.button("Run")

if 'image' not in st.session_state:
    st.session_state['image'] = None

model = load_model()
upscaler = load_upscaler()
if go:
    with st.spinner("Runing model..."):
        image = run(text, model)
        image = image.permute(0, 3, 1, 2)/255
        st.session_state.image = image

if st.session_state.image is not None:
    grid = utils.make_grid(st.session_state.image, nrow=2)
    out_image = grid.cpu().permute(1,2,0).clamp(0,1)*255
    display_image = Image.fromarray(out_image.to(torch.uint8).numpy())
    st.image(display_image)

if st.session_state.image is not None:
    if st.checkbox("upscale?"):
        factor = st.select_slider("Upscale factor", [1, 2, 4])
        with st.spinner("Runing model..."):
            image = upscale(st.session_state.image, upscaler, factor)

            grid = utils.make_grid(image, nrow=2)
            out_image = grid.cpu().permute(1,2,0).clamp(0,1)*255
            display_image = Image.fromarray(out_image.to(torch.uint8).numpy())
        st.image(display_image)