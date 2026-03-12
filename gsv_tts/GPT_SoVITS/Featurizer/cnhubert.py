from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

import torch.nn as nn
import torch
from ...config import Config


class CNHubert(nn.Module):
    def __init__(self, base_path, tts_config: Config):
        super().__init__()
        self.model = HubertModel.from_pretrained(base_path, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path, local_files_only=True)
        self.eval()
        self = self.to(tts_config.device)
        if tts_config.is_half: self = self.half()

    def forward(self, x):
        model_param = next(self.model.parameters())
        model_dtype = model_param.dtype
        model_device = model_param.device
        if isinstance(x, torch.Tensor):
            input_values = x
            if input_values.dim() == 1:
                input_values = input_values.unsqueeze(0)
            elif input_values.dim() > 2:
                input_values = input_values.squeeze()
                if input_values.dim() == 1:
                    input_values = input_values.unsqueeze(0)
            input_values = input_values.to(x.device, dtype=model_dtype)
        else:
            input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(
                model_device, dtype=model_dtype
            )
        feats = self.model(input_values)["last_hidden_state"]
        return feats