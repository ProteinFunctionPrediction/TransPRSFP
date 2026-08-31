import torch
from transformers import Trainer

from transformers.trainer_pt_utils import LengthGroupedSampler

class MLPResidueClassifierTrainer(Trainer):
    def __init__(self, encoder_model=None, encoder_model_is_fixed=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder_model = encoder_model
        self.encoder_model_is_fixed = encoder_model_is_fixed

    # TODO create a superclass of MLPResidueClassifierTrainer and GPT2LMHeadTrainer
    def _get_train_sampler(self):
        if self.args.group_by_length:
            lengths = [int(torch.as_tensor(x["prot_attention_mask"]).sum().item()) for x in self.train_dataset.data]
            return LengthGroupedSampler(batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps, lengths=lengths)

        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.encoder_model:
            if self.encoder_model_is_fixed:
                self.encoder_model.eval()
                with torch.no_grad():
                    last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
            else:
                last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
        else:
            last_hidden_state = inputs["last_hidden_states"]
                
        outputs = model(last_hidden_state=last_hidden_state, labels=inputs["labels"])

        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model_inputs = {
            "go_input_ids": inputs["go_input_ids"],
            "prot_input_ids": inputs["prot_input_ids"],
            "prot_attention_mask": inputs["prot_attention_mask"],
            "labels": inputs["labels"]
        }

        if "last_hidden_states" in inputs:
            model_inputs["last_hidden_states"] = inputs["last_hidden_states"]
        
        return super().prediction_step(
            model,
            model_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys
        )