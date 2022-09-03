import os
import logging
import urllib
from glob import glob

import lightning as L
import gradio as gr
from lightning.app.components.serve import ServeGradio
from PIL import Image
from rich.logging import RichHandler

from diffusion_app.diffusion_demo import Demo

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)
num_imgs = 8
base_url = "https://raw.githubusercontent.com/luminide/diffusion-demo/main/assets"
os.makedirs("resources", exist_ok=True)
for i in range(num_imgs):
    urllib.request.urlretrieve(f"{base_url}/sketch{i}.png", f"resources/sketch{i}.png")


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = [
        gr.Image(shape=(512, 512), type="pil", label="Input image"),
        gr.Textbox(lines=2, label="Prompt"),
        gr.Slider(minimum=0.6, maximum=0.9, value=0.8, label="Strength"),
        gr.Slider(minimum=5, maximum=60, value=50, step=5, label="Steps"),
    ]
    outputs = [
        gr.Image(shape=(512, 512), type="pil", label="Generated image"),
    ]
    examples = [[f"resources/sketch{i}.png", "cinematic bladerunner cityscape at night, trending on artstation, featured on pixiv, 8k, matte painting, hyper detailed, unreal engine 5, epic lighting, in sharp focus, vivid colors, high contrast, award winning photorealism"] for i in range(num_imgs)]
    enable_queue = True

    def __init__(self):
        super().__init__(parallel=True, cloud_compute=L.CloudCompute("cpu-medium"))

    def build_model(self) -> Demo:
        logger.info("loading model...")
        model = Demo()
        logger.info("built model!")
        return model

    def predict(self, image, prompt, strength, steps) -> Image.Image:
        return self.model.predict(image, prompt, strength, steps)
