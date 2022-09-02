import logging
import urllib
from glob import glob

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
for i in range(num_imgs):
    urllib.request.urlretrieve(f"{base_url}/sketch{i}.png", f"resources/sketch{i}.png")


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = gr.inputs.Image(type="pil")
    outputs = [
        gr.outputs.Image(label="generated image"),
    ]
    examples = [f"resources/sketch{i}.png" for i in range(num_imgs)]
    enable_queue = True

    def __init__(self):
        super().__init__(parallel=True)

    def build_model(self) -> Demo:
        logger.info("loading model...")
        model = Demo()
        logger.info("built model!")
        return model

    def predict(self, image: Image.Image) -> Image.Image:
        return self.model.predict(image)
