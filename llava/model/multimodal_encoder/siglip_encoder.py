import torch
import torch.nn as nn

from transformers import AutoModel
try:
    from transformers import AutoImageProcessor
except Exception:
    AutoImageProcessor = None
try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.has_cls_token = False

        if not delay_load:
            self.load_model()
        else:
            self.vision_tower = None
            self.image_processor = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        if AutoImageProcessor is not None:
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        elif AutoProcessor is not None:
            processor = AutoProcessor.from_pretrained(self.vision_tower_name)
            self.image_processor = getattr(processor, "image_processor", processor)
        else:
            raise ImportError("AutoImageProcessor/AutoProcessor not available in this transformers version.")
        if not hasattr(self.image_processor, "preprocess") and hasattr(self.image_processor, "__call__"):
            self.image_processor.preprocess = self.image_processor.__call__
        self.vision_tower = AutoModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
        ).vision_model
        self.vision_tower.requires_grad_(False)
        self.has_cls_token = self._detect_cls_token()
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            if self.has_cls_token:
                image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    pixel_values=image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_out = self.vision_tower(
                pixel_values=images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True,
            )
            image_features = self.feature_select(image_forward_out).to(images.dtype)

        return image_features

    def _detect_cls_token(self):
        vision = self.vision_tower
        if hasattr(vision, "embeddings"):
            emb = vision.embeddings
            if hasattr(emb, "class_embedding") or hasattr(emb, "cls_token"):
                return True
            if hasattr(emb, "use_cls_token"):
                return bool(emb.use_cls_token)
        return False

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def hidden_size(self):
        for attr in ("hidden_size", "vision_hidden_size"):
            if hasattr(self.vision_tower.config, attr):
                return getattr(self.vision_tower.config, attr)
        if hasattr(self.vision_tower, "vision_model") and hasattr(self.vision_tower.vision_model, "config"):
            for attr in ("hidden_size", "vision_hidden_size"):
                if hasattr(self.vision_tower.vision_model.config, attr):
                    return getattr(self.vision_tower.vision_model.config, attr)
        raise AttributeError("SiglipVisionTower missing hidden_size in config.")
