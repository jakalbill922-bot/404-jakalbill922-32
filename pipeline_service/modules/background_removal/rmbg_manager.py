from __future__ import annotations

import time
import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop
from config import Settings
from logger_config import logger

from transformers import AutoModelForImageSegmentation


class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService with BiRefNet.
        """
        self.settings = settings

        # Set padding percentage and output size for centering and resizing
        self.padding_percentage = self.settings.padding_percentage
        self.limit_padding = self.settings.limit_padding
        self.output_size = self.settings.output_image_size

        # Set device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # BiRefNet model
        self.model = None
        self.model_id = "ZhengPeng7/BiRefNet"

        # Set transform for BiRefNet
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Set transform for output
        self.output_transform = transforms.Compose([
            transforms.Resize(self.settings.input_image_size), 
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService with BiRefNet.
        """
        logger.info(f"Loading BiRefNet model...")

        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            # BiRefNet is small enough to stay on GPU
            self.model.to(self.device).eval()
            logger.success(f"BiRefNet model loaded on {self.device}.")
        except Exception as e:
            logger.error(f"Error loading BiRefNet model: {e}")
            raise RuntimeError(f"Error loading BiRefNet model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def to_device(self, device: str) -> None:
        """Move model to specified device for offloading."""
        if self.model is not None:
            self.model.to(device)
            self.device = device
            logger.info(f"BiRefNet moved to {device}")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError("BiRefNet model not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image using BiRefNet.
        """
        t1 = time.time()
        
        # Check if the image has alpha channel
        has_alpha = False
        
        if image.mode == "RGBA":
            alpha = np.array(image)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        
        if has_alpha:
            output = image
            image_without_background = image
        else:
            # Convert to RGB
            rgb_image = image.convert('RGB')
            original_size = rgb_image.size
            
            # Get mask from BiRefNet
            mask = self._get_mask(rgb_image)
            
            # Resize mask to original size
            mask_pil = to_pil_image(mask)
            mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)
            
            # Apply mask
            rgb_image_resized = rgb_image.resize(self.settings.input_image_size)
            mask_resized = mask_pil.resize(self.settings.input_image_size)
            
            # Create RGBA image
            rgba = rgb_image_resized.convert("RGBA")
            rgba.putalpha(mask_resized)
            
            # Convert to tensor for cropping
            foreground_tensor = self.output_transform(rgba.convert('RGB'))
            mask_tensor = transforms.ToTensor()(mask_resized)
            combined = torch.cat([foreground_tensor, mask_tensor], dim=0)
            
            output = self._crop_and_center(combined)
            image_without_background = to_pil_image(output[:3])

        removal_time = time.time() - t1
        logger.success(f"Background remove (BiRefNet) - Time: {removal_time:.2f}s - OutputSize: {image_without_background.size} - InputSize: {image.size}")

        return image_without_background

    def _get_mask(self, image: Image.Image) -> torch.Tensor:
        """
        Get segmentation mask from BiRefNet.
        """
        # Prepare input
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid()
        
        # Get mask
        mask = preds[0].squeeze()
        return mask.cpu()

    def _crop_and_center(self, foreground_tensor: torch.Tensor) -> torch.Tensor:
        """
        Crop and center the foreground object.
        """
        tensor_rgb = foreground_tensor[:3]
        mask = foreground_tensor[-1]

        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > 0.8)
        logger.info(f"BBOX len: {len(bbox_indices)}")
        
        if len(bbox_indices) == 0:
            crop_args = dict(top=0, left=0, height=mask.shape[0], width=mask.shape[1])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 0])
            w_min, w_max = torch.aminmax(bbox_indices[:, 1])
            height, width = h_max - h_min, w_max - w_min
            center = ((h_max + h_min) / 2, (w_max + w_min) / 2)
            size = max(width, height)
            padded_size_factor = 1 + self.padding_percentage
            size = int(size * padded_size_factor)

            top = int(center[0] - size // 2)
            left = int(center[1] - size // 2)
            bottom = int(center[0] + size // 2)
            right = int(center[1] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[0], bottom)
                right = min(mask.shape[1], right)
            
            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left
            )

        logger.info(f"CROP: {crop_args}")
        mask = mask.unsqueeze(0)
        tensor_rgba = torch.cat([tensor_rgb * mask, mask], dim=-3)
        output = resized_crop(tensor_rgba, **crop_args, size=self.output_size, antialias=False)
        return output
