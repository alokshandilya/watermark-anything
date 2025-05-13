import json
import os
import omegaconf
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union

from watermark_anything.models import Wam, build_embedder, build_extractor
from watermark_anything.augmentation.augmenter import Augmenter
from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img
from watermark_anything.modules.jnd import JND

def str2msg(string):
    """Convert a binary string to a list of booleans."""
    return [True if el=='1' else False for el in string]

def msg2str(msg):
    """Convert a list of booleans to a binary string."""
    return "".join([('1' if el else '0') for el in msg])

class Watermarker:
    """
    A wrapper class around Meta's Wam implementation to provide a simplified interface
    for watermarking images.
    """
    def __init__(self, device=None, checkpoint_path=None, params_path=None):
        """
        Initialize the watermarker with the given device and checkpoint path.
        
        Args:
            device: The device to use (CPU or CUDA)
            checkpoint_path: Path to the checkpoint file
            params_path: Path to the JSON file containing the parameters
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = default_transform  # Don't call the transform, just store it
        
        # If checkpoint_path is provided, load the model
        if checkpoint_path is not None:
            self.load_model(checkpoint_path, params_path)
    
    def load_model(self, checkpoint_path, params_path=None):
        """
        Load a model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            params_path: Path to the JSON file containing the parameters (optional)
        """
        # If params_path is not provided, try to find it in the same directory
        if params_path is None:
            params_path = os.path.join(os.path.dirname(checkpoint_path), "params.json")
            # If it doesn't exist, use default parameters
            if not os.path.exists(params_path):
                # Default parameters for WAM
                params = {
                    "embedder_config": os.path.join(os.path.dirname(__file__), "../configs/embedder.yaml"),
                    "extractor_config": os.path.join(os.path.dirname(__file__), "../configs/extractor.yaml"),
                    "augmentation_config": os.path.join(os.path.dirname(__file__), "../configs/all_augs.yaml"),
                    "attenuation_config": os.path.join(os.path.dirname(__file__), "../configs/attenuation.yaml"),
                    "embedder_model": "unet",
                    "extractor_model": "resnet18",
                    "attenuation": "default",
                    "nbits": 32,
                    "img_size": 224,
                    "scaling_w": 0.01,
                    "scaling_i": 0.1
                }
                # Create a temp params file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(params, f)
                    params_path = f.name
        
        try:
            # Load the parameters
            with open(params_path, 'r') as file:
                params = json.load(file)
            
            # Create an argparse Namespace object from the parameters
            from argparse import Namespace
            args = Namespace(**params)
            
            # Load configurations
            embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
            embedder_params = embedder_cfg[args.embedder_model]
            extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
            extractor_params = extractor_cfg[args.extractor_model]
            augmenter_cfg = omegaconf.OmegaConf.load(args.augmentation_config)
            attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
            
            # Build models
            embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
            extractor = build_extractor(extractor_cfg.model, extractor_params, args.img_size, args.nbits)
            augmenter = Augmenter(**augmenter_cfg)
            
            try:
                attenuation = JND(**attenuation_cfg[args.attenuation], preprocess=unnormalize_img, postprocess=normalize_img)
            except:
                attenuation = None
            
            # Build the complete model
            self.model = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)
            self.model.to(self.device)
            
            # Load the model weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Watermark model loaded successfully from {checkpoint_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Load the model state dict.
        
        Args:
            state_dict: The state dict to load
        """
        if self.model is None:
            # Initialize a default model if none exists
            from argparse import Namespace
            args = Namespace(
                embedder_model="unet",
                extractor_model="resnet18",
                attenuation="default",
                nbits=32,
                img_size=224,
                scaling_w=0.01,
                scaling_i=0.1
            )
            
            # Get config paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
            config_dir = os.path.join(base_dir, "configs")
            
            # Load configurations
            embedder_cfg = omegaconf.OmegaConf.load(os.path.join(config_dir, "embedder.yaml"))
            embedder_params = embedder_cfg[args.embedder_model]
            extractor_cfg = omegaconf.OmegaConf.load(os.path.join(config_dir, "extractor.yaml"))
            extractor_params = extractor_cfg[args.extractor_model]
            augmenter_cfg = omegaconf.OmegaConf.load(os.path.join(config_dir, "all_augs.yaml"))
            attenuation_cfg = omegaconf.OmegaConf.load(os.path.join(config_dir, "attenuation.yaml"))
            
            # Build models
            embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
            extractor = build_extractor(extractor_cfg.model, extractor_params, args.img_size, args.nbits)
            augmenter = Augmenter(**augmenter_cfg)
            
            try:
                attenuation = JND(**attenuation_cfg[args.attenuation], preprocess=unnormalize_img, postprocess=normalize_img)
            except:
                attenuation = None
            
            # Build the complete model
            self.model = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)
            self.model.to(self.device)
        
        # Load the state dict
        self.model.load_state_dict(state_dict)
    
    def eval(self):
        """Set the model to evaluation mode."""
        if self.model:
            self.model.eval()
        return self
    
    def create_mask(self, image_tensor, mask=None):
        """
        Create a mask for the watermark if none is provided.
        
        Args:
            image_tensor: The image tensor
            mask: Optional mask tensor
        
        Returns:
            A mask tensor
        """
        # If mask is not provided, create a full image mask
        if mask is None:
            B, C, H, W = image_tensor.shape
            mask = torch.ones((B, 1, H, W), device=self.device)
        return mask

    def preprocess_image(self, image):
        """
        Preprocess an image for the model.
        
        Args:
            image: A PIL Image or a tensor
        
        Returns:
            A preprocessed tensor
        """
        if isinstance(image, torch.Tensor):
            # If already a tensor, ensure it's in the right format
            if image.dim() == 3:  # CHW
                image = image.unsqueeze(0)  # Add batch dimension
            if image.shape[1] == 4:  # RGBA
                # Convert RGBA to RGB by removing alpha channel
                image = image[:, :3]
            return image.to(self.device)
        else:
            # Convert PIL Image to tensor
            if image.mode == 'RGBA':
                # Convert RGBA to RGB by removing alpha channel
                image = image.convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor

    def postprocess_image(self, tensor):
        """
        Convert a tensor back to a PIL Image.
        
        Args:
            tensor: The image tensor
        
        Returns:
            A PIL Image
        """
        # Convert the tensor to a numpy array
        tensor = tensor.squeeze(0).cpu()
        if tensor.shape[0] == 3:  # If it's a RGB tensor
            # Unnormalize and clamp
            tensor = unnormalize_img(tensor).clamp(0, 1)
            # Convert to numpy array
            array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Convert to PIL Image
            return Image.fromarray(array)
        else:
            raise ValueError("Expected a RGB tensor")

    def encode(self, image, watermark_id, MODE='binary'):
        """
        Embed a watermark in an image.
        
        Args:
            image: A PIL Image or a tensor
            watermark_id: The watermark ID (binary string or tensor)
            MODE: The mode of watermark encoding ('binary' or other)
        
        Returns:
            A watermarked PIL Image
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert watermark_id to the right format
        if MODE == 'binary':
            if isinstance(watermark_id, str):
                # Convert binary string to list of booleans
                wm = str2msg(watermark_id)
                # Convert to tensor
                watermark = torch.tensor(wm, dtype=torch.bool).to(self.device)
            else:
                # Assume it's already a tensor of the right format
                watermark = watermark_id.to(self.device)
        else:
            raise ValueError(f"Unsupported encoding mode: {MODE}")
        
        # Preprocess the image
        img_tensor = self.preprocess_image(image)
        
        # Generate a full image mask
        mask = self.create_mask(img_tensor)
        
        # Apply the watermark
        with torch.no_grad():
            watermarked_tensor = self.model.watermark(img_tensor, mask, watermark)
        
        # Convert back to PIL Image
        return self.postprocess_image(watermarked_tensor)
    
    def decode(self, image, mask=None, threshold=0.5):
        """
        Detect and extract a watermark from an image.
        
        Args:
            image: A PIL Image or a tensor
            mask: Optional mask tensor
            threshold: Threshold for watermark detection
        
        Returns:
            (watermark_id, is_present, confidence)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Preprocess the image
        img_tensor = self.preprocess_image(image)
        
        # Create or use mask
        mask_tensor = self.create_mask(img_tensor, mask)
        
        # Detect the watermark
        with torch.no_grad():
            wm_pred, _, conf = self.model.extract(img_tensor, mask_tensor, softmax=True)
        
        # Convert to binary string
        watermark_id = msg2str(wm_pred.cpu().squeeze().tolist())
        
        # Check if watermark is present
        is_present = conf.mean().item() > threshold
        
        return watermark_id, is_present, conf.mean().item()
    
    def to(self, device):
        """
        Move the model to the specified device.
        
        Args:
            device: The device to move the model to
        
        Returns:
            self
        """
        self.device = device
        if self.model:
            self.model.to(device)
        return self
