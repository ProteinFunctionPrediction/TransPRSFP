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
        y_input = torch.tensor([SOS_token], dtype=torch.long, device=self.device)
        
        probs = None
        if return_probs:
            probs = []

        

        for _ in range(max_length):
            encoder_attention_mask = input_attention_mask[:, :last_hidden_state.shape[1]]
            pred = model(input_ids=y_input, encoder_hidden_states=last_hidden_state, encoder_attention_mask=encoder_attention_mask)
            next_token_logits = pred.logits[-1, :]
            next_token_logits[SOS_token] = -1e20
            next_token_logits[EOS_token] = -1e20
            next_token_logits[Settings.TRANSFORMER_TRG_PAD_IDX] = -1e20
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            if return_probs:
                softmax = torch.nn.Softmax()
                next_token_probs = softmax(next_token_logits)
                topk_values, topk_indices = torch.topk(next_token_probs, k=keep_top)
                topk_values = topk_values.cpu().numpy()
                topk_indices = topk_indices.cpu().numpy()
                probs.append((topk_values, topk_indices))

            y_input = torch.cat((y_input, next_token), dim=-1)
        
        if return_probs:
            return [i for i in y_input.view(-1).tolist()][1:], probs
        else:
            return [i for i in y_input.view(-1).tolist()][1:]
