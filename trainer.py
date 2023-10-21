import torch
import copy

from typing import Dict, List, Optional
from transformers import Seq2SeqTrainer

class ToxicTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = self.model(
            input_ids=inputs['query_inputs'],
            attention_mask=inputs['query_masks'],
            labels=inputs['target_inputs'],
            output_attentions=False,
            output_hidden_states=False,
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model, 
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
            
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        generation_config = copy.deepcopy(self.model.generation_config)
        
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        
        generated_tokens = self.model.generate(inputs['query_inputs'], attention_mask=inputs['query_masks'], generation_config=generation_config)
        
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False
        
        gen_config = self.model.generation_config
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        loss = None
        if self.args.prediction_loss_only:
            return loss, None, None
        
        if "labels" in inputs:        
            labels = inputs["labels"]
        else:
            labels = inputs["target_inputs"]
        
        return loss, generated_tokens, labels