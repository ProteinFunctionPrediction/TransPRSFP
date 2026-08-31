from utils.model.residue_classifiers_common.residue_classifier_utils import ResidueClassifierUtils
from universal.settings.settings import Settings
import numpy as np
import torch

class GPT2LMHeadUtils(ResidueClassifierUtils):
    def __init__(self, device) -> None:
        super().__init__(device)

    def predict(self,
                encoder,
                model,
                input_sequence,
                input_attention_mask,
                max_length,
                SOS_token,
                EOS_token,
                EMPTY_token,
                OOV_token,
                return_probs=False,
                keep_top=10,
                embeddings=None,
                embedding_offset=None,
                embedding_length=None):
        model.eval()
        last_hidden_state = self.get_last_hidden_state(encoder,
                                                       input_sequence,
                                                       input_attention_mask,
                                                       embeddings,
                                                       embedding_offset,
                                                       embedding_length)

        encoder_attention_mask = input_attention_mask[:, :last_hidden_state.shape[1]]
        current_input = torch.tensor([SOS_token], dtype=torch.long, device=self.device)
        
        probs = None
        if return_probs:
            probs = []

        past_key_values = None
        prediction = []



        with torch.inference_mode():
            for _ in range(max_length):
                pred = model(input_ids=current_input,
                             past_key_values=past_key_values,
                             use_cache=True,
                             encoder_hidden_states=last_hidden_state,
                             encoder_attention_mask=encoder_attention_mask)
                next_token_logits = pred.logits[-1, :]
                next_token_logits[SOS_token] = -1e20
                next_token_logits[EOS_token] = -1e20
                next_token_logits[Settings.TRANSFORMER_TRG_PAD_IDX] = -1e20
                next_token = torch.argmax(next_token_logits, dim=-1)

                prediction.append(int(next_token.item()))

                if return_probs:
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    topk_values, topk_indices = torch.topk(next_token_probs, k=keep_top)
                    topk_values = topk_values.cpu().numpy()
                    topk_indices = topk_indices.cpu().numpy()
                    probs.append((topk_values, topk_indices))

                current_input = next_token.unsqueeze(-1)
                past_key_values = pred.past_key_values

        if return_probs:
            return prediction, probs
        else:
            return prediction
