# coding=utf-8
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch mT5 model."""

from ...utils import logging
from ..t5.modeling_t5 import T5EncoderModel, T5ForConditionalGeneration, T5Model
from .configuration_mt5 import MT5Config
import numpy as np
import torch


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"


class MT5Model(T5Model):
    r"""
    This class overrides [`T5Model`]. Please check the superclass for the appropriate documentation alongside usage
    examples.

    Examples:

    ```python
    >>> from transformers import MT5Model, T5Tokenizer

    >>> model = MT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> labels = tokenizer(text_target=summary, return_tensors="pt")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]


class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    r"""
    This class overrides [`T5ForConditionalGeneration`]. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples:

    ```python
    >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer

    >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]


    def __init__(self, config):
        super().__init__(config)
        # source language is English
        self.src_lang = 'en'


    def activate_adapter_projection_stack(self, adapter_name: str, layer_i: int, lang: str, proj_prob: float, encoder=True, decoder=True): 
        if encoder:
            self.encoder.block[layer_i].layer[-1].stack_projection_flag = True
            self.encoder.block[layer_i].layer[-1].task_adapter = adapter_name
            self.encoder.block[layer_i].layer[-1].proj_lang = lang
            self.encoder.block[layer_i].layer[-1].proj_prob = proj_prob
            self.encoder.block[layer_i].layer[-1].src_lang = self.src_lang
        if decoder:
            self.decoder.block[layer_i].layer[-1].stack_projection_flag = True
            self.decoder.block[layer_i].layer[-1].task_adapter = adapter_name
            self.decoder.block[layer_i].layer[-1].proj_lang = lang
            self.decoder.block[layer_i].layer[-1].proj_prob = proj_prob
            self.decoder.block[layer_i].layer[-1].src_lang = self.src_lang


    def activate_csi(self, adapter_name: str, layer_i: int, lang: str, encoder=True, decoder=True): # proj_prob: float):
        if encoder:
            self.encoder.block[layer_i].layer[-1].csi_flag = True
            self.encoder.block[layer_i].layer[-1].task_adapter = adapter_name
            self.encoder.block[layer_i].layer[-1].proj_lang = lang
            #self.encoder.block[layer_i].layer[-1].proj_prob = proj_prob
            self.encoder.block[layer_i].layer[-1].src_lang = self.src_lang
        if decoder:
            self.decoder.block[layer_i].layer[-1].csi_flag = True
            self.decoder.block[layer_i].layer[-1].task_adapter = adapter_name
            self.decoder.block[layer_i].layer[-1].proj_lang = lang
            #self.decoder.block[layer_i].layer[-1].proj_prob = proj_prob
            self.decoder.block[layer_i].layer[-1].src_lang = self.src_lang


    def disable_csi(self):
        for layer_i in range(self.config.num_hidden_layers):
            if self.encoder.block[layer_i].layer[-1].csi_flag:
                self.encoder.block[layer_i].layer[-1].csi_flag = False
            if self.decoder.block[layer_i].layer[-1].csi_flag:
                self.decoder.block[layer_i].layer[-1].csi_flag = False


    def disable_adapter_projection_stack(self):
        for layer_i in range(self.config.num_hidden_layers):
            if self.encoder.block[layer_i].layer[-1].stack_projection_flag:
                self.encoder.block[layer_i].layer[-1].stack_projection_flag = False
            if self.decoder.block[layer_i].layer[-1].stack_projection_flag:
                self.decoder.block[layer_i].layer[-1].stack_projection_flag = False


    def load_adapter_projections(self, lang_list: list, variance_accounted: float, subspace_dir: str):
        if subspace_dir[-1] != '/': subspace_dir += '/'

        num_layers = self.config.num_hidden_layers
        dim_size = self.config.hidden_size

        # compute projections
        projection_dict = {}
        means_a_dict = {}
        means_b_dict = {}
        means_langs = {}

        for lang in lang_list:

            projections = []
            means_a = []
            means_b = []
            lang_m = []

            for layer_i in range(self.config.num_hidden_layers+1):
                mean_a = np.load(subspace_dir+self.src_lang+'_layer'+str(layer_i)+'_mean.npy') # change for other projections
                mean_b = np.load(subspace_dir+self.src_lang+'_layer'+str(layer_i)+'_mean.npy') # change for other projections
                #mean_b = np.load(subspace_dir+lang+'_layer'+str(layer_i)+'_mean.npy') # change for other projections

                means_a.append(mean_a)
                means_b.append(mean_b)

                s = np.load(subspace_dir+lang+'_layer'+str(layer_i)+'_s.npy')
                vh = np.load(subspace_dir+lang+'_layer'+str(layer_i)+'_vh.npy')
                subspace_m = np.load(subspace_dir+lang+'_layer'+str(layer_i)+'_mean.npy')

                lang_m.append(subspace_m)

                v = np.transpose(vh) # columns of V form the desired orthonormal basis

                subspace_dim = 0
                s_squared = np.square(s)
                total_variance = np.sum(s_squared) # Proportional to total variance.
                cutoff_variance = variance_accounted * total_variance
                curr_variance = 0.0
                for i in range(s.shape[-1]):
                    curr_variance += s_squared[i]
                    if curr_variance >= cutoff_variance:
                        subspace_dim = i+1
                        break
                # Projection matrix: convert into basis (excluding some dimensions), then
                # convert back into standard basis.
                v = v[:, :subspace_dim]
                projection_matrix = np.matmul(v, np.transpose(v))
                projections.append(projection_matrix)

                projection_dict[lang] = projections
                means_a_dict[lang] = means_a
                means_b_dict[lang] = means_b

            means_langs[lang] = lang_m

        self.set_adapter_projections(projection_dict, lang_list, means_a_dict, means_b_dict, means_langs)


    def set_adapter_projections(self, projection_dict, lang_list, means_a_dict, means_b_dict, means_langs):
        for lang in lang_list:
            for layer_i in range(1, self.config.num_hidden_layers+1):
                projection, projection_shift = self.compute_projection(projection_dict, means_a_dict, means_b_dict, lang, layer_i)
                self.encoder.block[layer_i-1].layer[-1].projections[lang] = projection
                self.encoder.block[layer_i-1].layer[-1].projections_shifts[lang] = projection_shift
                self.encoder.block[layer_i-1].layer[-1].lang_means[lang] = torch.from_numpy(means_langs[lang][layer_i])

                self.decoder.block[layer_i-1].layer[-1].projections[lang] = projection
                self.decoder.block[layer_i-1].layer[-1].projections_shifts[lang] = projection_shift
                self.decoder.block[layer_i-1].layer[-1].lang_means[lang] = torch.from_numpy(means_langs[lang][layer_i])
                # add shifts here


    def compute_projection(self, projection_dict, means_a_dict, means_b_dict, lang, layer_i):
        dim_size = self.config.hidden_size
        projection = torch.tensor(projection_dict[lang][layer_i]).float()
        mean_a = torch.tensor(means_a_dict[lang][layer_i]).float()
        mean_b = torch.tensor(means_b_dict[lang][layer_i]).float()
        projection_shift = mean_b - torch.matmul(projection, mean_a)
        projection_shift = projection_shift.reshape(1, 1, dim_size)
        return projection, projection_shift


class MT5EncoderModel(T5EncoderModel):
    r"""
    This class overrides [`T5EncoderModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.

    Examples:

    ```python
    >>> from transformers import MT5EncoderModel, T5Tokenizer

    >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```"""

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]
