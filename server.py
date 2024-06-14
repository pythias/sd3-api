import logging
import os
import sys
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Union, Any

from PIL import Image, ImageOps

import fire
from flask import Flask, jsonify, request

import torch
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from diffusers.image_processor import PipelineImageInput, is_valid_image
from diffusers.utils import load_image


def fix_image(image: Image.Image):
    if image is None:
        return None

    try:
        image = ImageOps.exif_transpose(image)
        image = fix_png_transparency(image)
    except Exception:
        pass

    return image


def fix_png_transparency(image: Image.Image):
    if image.mode not in ("RGB", "P") or not isinstance(image.info.get("transparency"), bytes):
        return image

    image = image.convert("RGBA")
    return image


class Processing(object):
    def __init__(
        self,
        prompt: Union[str, List[str]] = "",
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        return_dict: bool = True,
        clip_skip: Optional[int] = None,
    ):
        self.prompt = prompt
        self.prompt_2 = prompt_2
        self.prompt_3 = prompt_3
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.negative_prompt_2 = negative_prompt_2
        self.negative_prompt_3 = negative_prompt_3
        self.num_images_per_prompt = num_images_per_prompt
        self.return_dict = return_dict
        self.clip_skip = clip_skip


class ProcessingText2Image(Processing):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        self.width = width
        self.height = height


class ProcessingImage2Image(Processing):
    def __init__(self, image: Union[PipelineImageInput, str] = None, strength: float = 0.6, **kwargs):
        super().__init__(**kwargs)

        self.image = image
        self.strength = strength


class Server(object):
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        dtype=torch.float16,
        device: int = 2,
        host: str = "127.0.0.1",
        port: int = 9003,
        output_dir: str = "/tmp",
        output_host: str = "http://127.0.0.1",
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.host = host
        self.port = port
        self.lock = threading.Lock()

        self.output_dir = output_dir
        self.output_host = output_host

        if self.output_host.endswith("/"):
            self.output_host = self.output_host[:-1]

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
        )

    def load_model(self):
        self.pipe = DiffusionPipeline.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.components = self.pipe.components

        self.t2i_pipe = StableDiffusion3Pipeline(**self.components)
        self.t2i_pipe = self.t2i_pipe.to(f"cuda:{self.device}")

        self.i2i_pipe = StableDiffusion3Img2ImgPipeline(**self.components)
        self.i2i_pipe = self.i2i_pipe.to(f"cuda:{self.device}")

    def run(self):
        self.setup_logging()

        logging.info("Loading model...")
        self.load_model()

        logging.info("Warming up the model...")
        self.warmup()

        logging.info("Starting server...")
        app = Flask(__name__)

        @app.get("/hello")
        def hello():
            return "Hello, World!"

        @app.post("/t2i")
        def t2i():
            params = self.to_params(request)
            p = ProcessingText2Image(**params)
            return jsonify(self.text2image(p))

        @app.post("/i2i")
        def i2i():
            params = self.to_params(request)
            p = ProcessingImage2Image(**params)
            return jsonify(self.image2image(p))

        app.json.ensure_ascii = False
        app.run(port=self.port, host=self.host)

        logging.info("Server started at %s:%d", self.host, self.port)

    def to_params(self, request):
        # if request was json, use request.json
        if request.json:
            return request.json

        # if request was form, use request.form
        return request.form.to_dict()

    def warmup(self):
        self.t2i_pipe("A cat holding a sign that says hello world", num_inference_steps=28, guidance_scale=7.0).images[0]
        logging.info("Model finished loading, ready to start.")

    def text2image(self, p: ProcessingText2Image):
        with self.lock:
            images = self.t2i_pipe(**p.__dict__).images

        return self.to_out_images(images)

    def image2image(self, p: ProcessingImage2Image):
        loaded_image = load_image(p.image)
        if loaded_image is None:
            return {"error": "Invalid input image"}

        loaded_image = fix_image(loaded_image)

        p.image = loaded_image
        with self.lock:
            images = self.i2i_pipe(**p.__dict__).images

        return self.to_out_images(images)

    def to_out_images(self, images):
        if not images:
            return {"error": "No images generated"}

        image_urls = []
        for image in images:
            image_urls.append(self.save_image(image))
        return {"images": image_urls}

    def save_image(self, image):
        name = os.path.join(time.strftime("%Y%m%d"), str(uuid.uuid4()) + ".png")
        path = os.path.join(self.output_dir, name)
        url = self.output_host + "/" + name

        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

        logging.info("image saved, url: %s, path: %s", url, path)
        return url


if __name__ == "__main__":
    fire.Fire(Server)
