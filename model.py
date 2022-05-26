from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Union

import dlib
import huggingface_hub
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, 'encoder4editing')

from models.psp import pSp
from utils.alignment import align_face

sys.path.insert(0, 'HairCLIP/')
sys.path.insert(0, 'HairCLIP/mapper/')

from mapper.datasets.latents_dataset_inference import LatentsDatasetInference
from mapper.hairclip_mapper import HairCLIPMapper

TOKEN = os.environ['TOKEN']


class Model:
    def __init__(self, device: Union[torch.device, str]):
        self.device = torch.device(device)
        self.landmark_model = self._create_dlib_landmark_model()
        self.e4e = self._load_e4e()
        self.hairclip = self._load_hairclip()
        self.transform = self._create_transform()

    @staticmethod
    def _create_dlib_landmark_model():
        path = huggingface_hub.hf_hub_download(
            'hysts/dlib_face_landmark_model',
            'shape_predictor_68_face_landmarks.dat',
            use_auth_token=TOKEN)
        return dlib.shape_predictor(path)

    def _load_e4e(self) -> nn.Module:
        ckpt_path = huggingface_hub.hf_hub_download('hysts/e4e',
                                                    'e4e_ffhq_encode.pt',
                                                    use_auth_token=TOKEN)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        opts = ckpt['opts']
        opts['device'] = self.device.type
        opts['checkpoint_path'] = ckpt_path
        opts = argparse.Namespace(**opts)
        model = pSp(opts)
        model.to(self.device)
        model.eval()
        return model

    def _load_hairclip(self) -> nn.Module:
        ckpt_path = huggingface_hub.hf_hub_download('hysts/HairCLIP',
                                                    'hairclip.pt',
                                                    use_auth_token=TOKEN)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        opts = ckpt['opts']
        opts['device'] = self.device.type
        opts['checkpoint_path'] = ckpt_path
        opts['editing_type'] = 'both'
        opts['input_type'] = 'text'
        opts['hairstyle_description'] = 'HairCLIP/mapper/hairstyle_list.txt'
        opts['color_description'] = 'red'
        opts = argparse.Namespace(**opts)
        model = HairCLIPMapper(opts)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _create_transform() -> Callable:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return transform

    def detect_and_align_face(self, image) -> PIL.Image.Image:
        image = align_face(filepath=image.name, predictor=self.landmark_model)
        return image

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp((tensor + 1) / 2 * 255, 0, 255).to(torch.uint8)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = self.denormalize(tensor)
        return tensor.cpu().numpy().transpose(1, 2, 0)

    @torch.inference_mode()
    def reconstruct_face(
            self, image: PIL.Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        input_data = self.transform(image).unsqueeze(0).to(self.device)
        reconstructed_images, latents = self.e4e(input_data,
                                                 randomize_noise=False,
                                                 return_latents=True)
        reconstructed = torch.clamp(reconstructed_images[0].detach(), -1, 1)
        reconstructed = self.postprocess(reconstructed)
        return reconstructed, latents[0]

    @torch.inference_mode()
    def generate(self, editing_type: str, hairstyle_index: int,
                 color_description: str, latent: torch.Tensor) -> np.ndarray:
        opts = self.hairclip.opts
        opts.editing_type = editing_type
        opts.color_description = color_description

        if editing_type == 'color':
            hairstyle_index = 0

        device = torch.device(opts.device)

        dataset = LatentsDatasetInference(latents=latent.unsqueeze(0).cpu(),
                                          opts=opts)
        w, hairstyle_text_inputs_list, color_text_inputs_list = dataset[0][:3]

        w = w.unsqueeze(0).to(device)
        hairstyle_text_inputs = hairstyle_text_inputs_list[
            hairstyle_index].unsqueeze(0).to(device)
        color_text_inputs = color_text_inputs_list[0].unsqueeze(0).to(device)

        hairstyle_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).to(device)
        color_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).to(device)

        w_hat = w + 0.1 * self.hairclip.mapper(
            w,
            hairstyle_text_inputs,
            color_text_inputs,
            hairstyle_tensor_hairmasked,
            color_tensor_hairmasked,
        )
        x_hat, _ = self.hairclip.decoder(
            [w_hat],
            input_is_latent=True,
            return_latents=True,
            randomize_noise=False,
            truncation=1,
        )
        res = torch.clamp(x_hat[0].detach(), -1, 1)
        res = self.postprocess(res)
        return res
