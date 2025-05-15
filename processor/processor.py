import os
import torch

from typing import Union, List
from PIL import Image
from transformers import ProcessorMixin, AutoProcessor, BatchFeature

from utils import ideal_low_pass_filter, ideal_high_pass_filter, gaussian_high_pass_filter, gaussian_low_pass_filter


class MFCDProcessor:
    def __init__(
            self,
            processor: ProcessorMixin,
            high_pass_cutoff: float = 120,
            low_pass_cutoff: float = 30,
            filter_type: str = "ideal",
            device: Union[str, torch.device] = "cpu",
    ):
        self.processor_inside = processor
        self.high_pass_cutoff = high_pass_cutoff
        self.low_pass_cutoff = low_pass_cutoff
        self.filter_type = filter_type
        self.high_pass_filter = ideal_high_pass_filter if self.filter_type == "ideal" else gaussian_high_pass_filter
        self.low_pass_filter = ideal_low_pass_filter if self.filter_type == "ideal" else gaussian_low_pass_filter
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        for attr_name in dir(self.processor_inside):
            if not attr_name.startswith("__") and attr_name not in [
                "from_pretrained"
            ]:
                if callable(getattr(self.processor_inside, attr_name)):
                    setattr(self, attr_name, self._create_proxy_method(getattr(self.processor_inside, attr_name)))
                else:
                    setattr(self, attr_name, getattr(self.processor_inside, attr_name))

    def _create_proxy_method(self, method):
        def proxy(*args, **kwargs):
            return method(*args, **kwargs)

        return proxy

    def __call__(
            self,
            images: Union[Image.Image, List[Image.Image]],
            text: Union[str, List[str]],
            **kwargs,
    ) -> BatchFeature:
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(text, str):
            text = [text]
        with torch.no_grad():
            high_pass_images = [
                self.high_pass_filter(
                    image=image,
                    cutoff=self.high_pass_cutoff,
                    device=self.device
                ) for image in images
            ]
            low_pass_images = [
                self.low_pass_filter(
                    image=image,
                    cutoff=self.low_pass_cutoff,
                    device=self.device
                ) for image in images
            ]
        text = text * 3
        images = [
            *images,
            *high_pass_images,
            *low_pass_images,
        ]
        if "padding" in kwargs:
            kwargs.pop("padding")
        inputs = self.processor_inside.__call__(
            images=images,
            text=text,
            padding="longest",
            **kwargs
        )
        return inputs

    @staticmethod
    def from_pretrained(
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs,
    ) -> "MFCDProcessor":
        device = kwargs.pop("device", "cpu")
        device = device if isinstance(device, torch.device) else torch.device(device)
        high_pass_cutoff = kwargs.pop("high_pass_cutoff", 120)
        low_pass_cutoff = kwargs.pop("low_pass_cutoff", 30)
        filter_type = kwargs.pop("filter_type", "ideal")
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs
        )
        return MFCDProcessor(
            processor=processor,
            high_pass_cutoff=high_pass_cutoff,
            low_pass_cutoff=low_pass_cutoff,
            filter_type=filter_type,
            device=device,
        )
