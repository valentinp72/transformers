# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class SpeechEncoderMultipleDecodersConfig(PretrainedConfig):
    r"""
    [`SpeechEncoderMultipleDecodersConfig`] is the configuration class to store the configuration of a
    [`SpeechEncoderMultipleDecodersModel`]. It is used to instantiate an Encoder Decoder model according to the specified
    arguments, defining the encoder and decoder configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderMultipleDecodersConfig, SpeechEncoderMultipleDecodersModel

    >>> # Initializing a Wav2Vec2 & BERT style configuration
    >>> config_encoder = Wav2Vec2Config()
    >>> config_decoder = BertConfig()

    >>> config = SpeechEncoderMultipleDecodersConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a Wav2Vec2Bert model from a Wav2Vec2 & bert-base-uncased style configurations
    >>> model = SpeechEncoderMultipleDecodersModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = SpeechEncoderMultipleDecodersConfig.from_pretrained("my-model")
    >>> model = SpeechEncoderMultipleDecodersModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""
    model_type = "speech-encoder-multiple-decoders"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        decoders_names = [kw for kw in kwargs if kw.startswith("decoder_")]
        if "encoder" not in kwargs or len(decoders_names) == 0:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because not both `encoder` and"
                f" `decoder_` sub-configurations are passed, but only {kwargs}"
            )

        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")

        decoders_configs = {}
        decoders_models_types = {}
        for decoder_name in decoders_names:
            decoders_configs[decoder_name] = kwargs.pop(decoder_name)
            decoders_models_types[decoder_name] = decoders_configs[decoder_name].pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoders = {}
        for decoder_name in decoders_names:
            self.decoders[decoder_name] = AutoConfig.for_model(decoders_models_types[decoder_name], **decoders_configs[decoder_name])
        self.decoders_names = decoders_names
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoders_configs: dict[PretrainedConfig], **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`SpeechEncoderMultipleDecodersConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`SpeechEncoderMultipleDecodersConfig`]: An instance of a configuration object
        """
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        for decoder_name in decoders_configs:
            decoders_configs[decoder_name].is_decoder = True
            decoders_configs[decoder_name].add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), **{decoder_name: decoders_configs[decoder_name].to_dict() for decoder_name in decoders_configs}, **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoders_names"] = list(self.decoders.keys())
        for decoder_name in self.decoders:
            output[decoder_name] = self.decoders[decoder_name].to_dict()
        del output["decoders"]
        output["model_type"] = self.__class__.model_type
        return output
