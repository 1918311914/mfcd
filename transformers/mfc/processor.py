import os
import torch

from typing import Union
from transformers import ProcessorMixin, AutoProcessor, BatchFeature
from transformers.image_utils import ImageInput

from utils import high_pass_filter, low_pass_filter


class MFCProcessor:
    def __init__(
            self,
            processor: ProcessorMixin,
            high_pass_cutoff: int = 60,
            low_pass_cutoff: int = 80,
            device: Union[str, torch.device] = "cpu",
    ):
        self.processor = processor
        self.high_pass_cutoff = high_pass_cutoff
        self.low_pass_cutoff = low_pass_cutoff
        self.device = device if isinstance(device, torch.device) else torch.device(device)

    def __getattr__(self, item):
        return getattr(self.processor, item)

    def __call__(
            self,
            images: ImageInput = None,
            **kwargs,
    ) -> BatchFeature:
        high_pass_images = high_pass_filter(
            image=images,
            cutoff=self.high_pass_cutoff,
            device=self.device
        )
        low_pass_images = low_pass_filter(
            image=images,
            cutoff=self.low_pass_cutoff,
            device=self.device
        )
        inputs = self.processor.__call__(
            images=images,
            **kwargs
        )
        pixel_values = inputs.pop("pixel_values")
        high_pass_pixel_values = self.processor.__call__(
            images=high_pass_images,
            **kwargs
        )["pixel_values"]
        low_pass_pixel_values = self.processor.__call__(
            images=low_pass_images,
            **kwargs
        )["pixel_values"]
        inputs["pixel_values"] = (pixel_values, high_pass_pixel_values, low_pass_pixel_values)
        return inputs

    @staticmethod
    def from_pretrained(
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs,
    ) -> "MFCProcessor":
        device = kwargs.pop("device", "cpu")
        device = device if isinstance(device, torch.device) else torch.device(device)
        high_pass_cutoff = kwargs.pop("high_pass_cutoff", 60)
        low_pass_cutoff = kwargs.pop("low_pass_cutoff", 80)
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs
        )
        return MFCProcessor(
            processor=processor,
            high_pass_cutoff=high_pass_cutoff,
            low_pass_cutoff=low_pass_cutoff,
            device=device,
        )
