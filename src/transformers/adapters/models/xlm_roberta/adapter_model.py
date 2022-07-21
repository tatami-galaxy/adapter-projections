from ....models.xlm_roberta.modeling_xlm_roberta import XLM_ROBERTA_START_DOCSTRING, XLMRobertaConfig
from ....utils import add_start_docstrings
from ..roberta.adapter_model import RobertaAdapterModel, RobertaModelWithHeads
import numpy as np
import torch
import copy


@add_start_docstrings(
    """XLM-RoBERTa Model with the option to add multiple flexible heads on top.""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaAdapterModel(RobertaAdapterModel):
    """
    This class overrides :class:`~transformers.RobertaAdapterModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        # source language is English
        self.src_lang = 'en'


    def activate_embedding_projection(self, lang):
        if not self.roberta.encoder.embedding_projection_flag:
            self.roberta.encoder.embedding_projection_flag = True
            self.roberta.encoder.proj_lang = lang


    def activate_adapter_projection_stack(self, adapter_name: str, layer_i: int, lang: str, prob: float):
        self.roberta.encoder.layer[layer_i].output.stack_projection_flag = True
        self.roberta.encoder.layer[layer_i].output.task_adapter = adapter_name
        self.roberta.encoder.layer[layer_i].output.proj_lang = lang
        self.roberta.encoder.layer[layer_i].output.prob = prob 
        self.roberta.encoder.layer[layer_i].output.src_lang = self.src_lang


    def activate_adapter_projection_parallel(self, task_adapter_name: str, parallel_adapter_name: str, lang: str):
        for layer_i in range(self.config.num_hidden_layers):
            self.roberta.encoder.layer[layer_i].output.parallel_projection_flag = True
            self.roberta.encoder.layer[layer_i].output.task_adapter = task_adapter_name
            self.roberta.encoder.layer[layer_i].output.parallel_adapter = parallel_adapter_name
            self.roberta.encoder.layer[layer_i].output.proj_lang = lang


    def disable_adapter_projection_stack(self):
        for layer_i in range(self.config.num_hidden_layers):
            if self.roberta.encoder.layer[layer_i].output.stack_projection_flag:
                self.roberta.encoder.layer[layer_i].output.stack_projection_flag = False


    def disable_adapter_projection_parallel(self):
        for layer_i in range(self.config.num_hidden_layers):
            if self.roberta.encoder.layer[layer_i].output.parallel_projection_flag:
                self.roberta.encoder.layer[layer_i].output.parallel_projection_flag = False


    def disable_embedding_projection(self):
        if self.roberta.encoder.embedding_projection_flag:
            self.roberta.encoder.embedding_projection_flag = False



    def load_adapter_projections(self, lang_list: list, variance_accounted: float, subspace_dir: str):
        if subspace_dir[-1] != '/': subspace_dir += '/'
        self.roberta.encoder.lang_list = lang_list

        num_layers = self.roberta.config.num_hidden_layers
        dim_size = self.roberta.config.hidden_size

        # compute projections
        projection_dict = {}
        means_a_dict = {}
        means_b_dict = {}

        for lang in lang_list:

            projections = []
            means_a = []
            means_b = []

            for layer_i in range(self.config.num_hidden_layers+1):
                mean_a = np.load(subspace_dir+self.src_lang+'_layer'+str(layer_i)+'_mean.npy') # change for other projections
                mean_b = np.load(subspace_dir+self.src_lang+'_layer'+str(layer_i)+'_mean.npy') # change for other projections

                means_a.append(mean_a)
                means_b.append(mean_b)

                s = np.load(subspace_dir+lang+'_layer'+str(layer_i)+'_s.npy')
                vh = np.load(subspace_dir+lang+'_layer'+str(layer_i)+'_vh.npy')
                subspace_m = np.load(subspace_dir+lang+'_layer'+str(layer_i)+'_mean.npy')

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

        self.set_adapter_projections(projection_dict, lang_list, means_a_dict, means_b_dict)


    def set_adapter_projections(self, projection_dict, lang_list, means_a_dict, means_b_dict):
        for lang in lang_list:
            projection, projection_shift = self.compute_projection(projection_dict, means_a_dict, means_b_dict, lang, 0) # 0 for embedding layer projections
            self.roberta.encoder.embedding_projections[lang] = projection #copy.deepcopy(projection)
            self.roberta.encoder.embedding_projections_shifts[lang] = projection_shift
            # add shifts here

        for lang in lang_list:
            for layer_i in range(1, self.config.num_hidden_layers+1):
                projection, projection_shift = self.compute_projection(projection_dict, means_a_dict, means_b_dict, lang, layer_i)
                self.roberta.encoder.layer[layer_i-1].output.projections[lang] = projection
                self.roberta.encoder.layer[layer_i-1].output.projections_shifts[lang] = projection_shift
                # add shifts here


    def compute_projection(self, projection_dict, means_a_dict, means_b_dict, lang, layer_i):
        dim_size = self.config.hidden_size
        projection = torch.tensor(projection_dict[lang][layer_i]).float() 
        mean_a = torch.tensor(means_a_dict[lang][layer_i]).float()
        mean_b = torch.tensor(means_b_dict[lang][layer_i]).float()
        projection_shift = mean_b - torch.matmul(projection, mean_a)
        projection_shift = projection_shift.reshape(1, 1, dim_size)
        return projection, projection_shift



@add_start_docstrings(
    """XLM-RoBERTa Model with the option to add multiple flexible heads on top.""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaModelWithHeads(RobertaModelWithHeads):
    """
    This class overrides :class:`~transformers.RobertaModelWithHeads`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
