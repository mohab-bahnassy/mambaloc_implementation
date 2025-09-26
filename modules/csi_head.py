# Copyright (c) 2024, Kevin Li, Aviv Bick.
# Adapted for CSI regression tasks

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from modules.uwb_backbone import UWBMixerModel
from utils.config import Config


@dataclass
class CSIRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_transfer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_mamba_outputs: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class CSIRegressionModel(nn.Module):
    def __init__(
        self, config: dict, initializer_cfg=None, device=None, dtype=None, **kwargs
    ) -> None:

        super().__init__()

        # Load config
        if not isinstance(config, Config):
            config = Config.from_dict(config)
        self.config = config

        # Factory kwargs
        factory_kwargs = {"device": device, "dtype": dtype}

        # CSI data dimensions
        input_features = config.CSIRegressionModel.input.input_features  # 540 for CSI data (270 magnitude + 270 phase)
        output_features = config.CSIRegressionModel.input.output_features  
        self.output_mode = config.CSIRegressionModel.input.output_mode  # "sequence" or "last"

        # CSI Mixer model (reuse UWB backbone architecture)
        self.backbone = UWBMixerModel(
            input_size=input_features,
            config=self.config,
            initializer_cfg=initializer_cfg,
            **factory_kwargs,
            **kwargs
        )

        # Regression head
        d_model = self.config.UWBMixerModel.input.d_model
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(config.CSIRegressionModel.input.get("dropout", 0.1))
        
        # Regression head - can predict for each timestep or just the last
        self.regression_head = nn.Linear(
            d_model, output_features, bias=True, **factory_kwargs
        )

        return

    def allocate_inference_cache(self, *args, **kwargs):
        return self.backbone.allocate_inference_cache(*args, **kwargs)

    def forward(
        self,
        input_data,
        targets=None,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        return_hidden_states=False,
        inference_params=None,
        position_ids=None,
    ):
        outputs = self.backbone(
            input_data,
            return_mixer_matrix=return_mixer_matrix,
            return_mamba_outputs=return_mamba_outputs,
            return_hidden_states=return_hidden_states,
            inference_params=inference_params,
            position_ids=position_ids,
        )

        predictions = None
        loss = None
        
        if outputs["last_hidden_state"] is not None:
            hidden_states = self.dropout(outputs["last_hidden_state"])
            
            # Get predictions
            predictions = self.regression_head(hidden_states)
            
            # Handle different output modes
            if self.output_mode == "last":
                predictions = predictions[:, -1, :]  # Only last timestep
            
            # Calculate loss if targets provided
            if targets is not None:
                if self.output_mode == "last" and targets.dim() == 3:
                    targets = targets[:, -1, :]  # Match prediction shape
                
                loss = nn.MSELoss()(predictions, targets)

        return CSIRegressionOutput(
            loss=loss,
            predictions=predictions,
            all_hidden_states=outputs["all_hidden_states"],
            all_transfer_matrices=outputs["all_transfer_matrices"],
            all_mamba_outputs=outputs["all_mamba_outputs"],
            last_hidden_state=outputs["last_hidden_state"],
        )

    def save_pretrained(self, save_directory):
        """
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f)

    @classmethod
    def from_pretrained(cls, model_path, config_path=None, **kwargs):
        """
        Load a pretrained model from a directory.
        """
        if config_path is None:
            config_path = os.path.join(model_path, "config.json")
        
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(config, **kwargs)
        
        # Load state dict
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model 