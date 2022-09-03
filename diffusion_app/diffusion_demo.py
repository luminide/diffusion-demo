import numpy as np
import requests
from PIL import Image

from openvino.runtime import Core
from diffusers import PNDMScheduler
from stable_diffusion_engine import StableDiffusionEngine


class Demo:
    def __init__(
            self, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"):
        scheduler = PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            skip_prk_steps=True,
            tensor_format="np"
        )

        self.stable_diffusion = StableDiffusionEngine(
            scheduler=scheduler,
        )
        print("Model loaded.")

    def predict(
            self, init_image, prompt, strength, steps,
            guidance_scale=7.5, eta=0.0):
        init_image = np.array(init_image)
        # convert from RGB to BGR
        init_image = init_image[:, :, ::-1]
        # run 
        print(f"Generating image from prompt: \n{prompt}")
        image = self.stable_diffusion(
            prompt=prompt, init_image=init_image,
            num_inference_steps=int(steps),
            strength=strength, guidance_scale=guidance_scale, eta=eta
        )

        # convert from BGR to RGB
        image = image[:, :, ::-1]
        return Image.fromarray(image)


if __name__ == "__main__":

    img_url = "https://raw.githubusercontent.com/luminide/diffusion-demo/main/assets/sketch0.png"
    img = Image.open(requests.get(img_url, stream=True).raw)
    model = Demo()
    result = model.predict(img)
    result.save('result.png')
