from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
try:
    from transformers import Qwen3MoeConfig, Qwen3MoeModel, Qwen3MoeForCausalLM
except Exception as exc:
    raise ImportError(
        "Qwen3 MoE requires transformers >= 4.51.0. "
        "Please upgrade transformers to load Qwen3-235B-A22B-Instruct-2507."
    ) from exc


class LlavaQwen3MoeConfig(Qwen3MoeConfig):
    model_type = "llava_qwen3_moe"


class LlavaQwen3MoeModel(LlavaMetaModel, Qwen3MoeModel):
    config_class = LlavaQwen3MoeConfig

    def __init__(self, config: Qwen3MoeConfig):
        super(LlavaQwen3MoeModel, self).__init__(config)


class LlavaQwen3MoeForCausalLM(Qwen3MoeForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen3MoeConfig

    def __init__(self, config):
        Qwen3MoeForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen3_moe"
        self.model = LlavaQwen3MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        ecgs: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                ecgs,
                images,
                image_sizes,
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        ecgs: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                ecgs,
                images,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            ecgs=ecgs,
            images=images,
            image_sizes=image_sizes,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        ecgs = kwargs.pop("ecgs", None)
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if ecgs is not None:
            inputs["ecgs"] = ecgs
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("llava_qwen3_moe", LlavaQwen3MoeConfig)
AutoModelForCausalLM.register(LlavaQwen3MoeConfig, LlavaQwen3MoeForCausalLM)
