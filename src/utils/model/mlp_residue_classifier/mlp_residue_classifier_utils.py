from universal.settings.settings import Settings
from utils.model.residue_classifiers_common.residue_classifier_utils import ResidueClassifierUtils
import torch

class MLPResidueClassifierUtils(ResidueClassifierUtils):
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


        with torch.no_grad():
            outputs = model(last_hidden_state=last_hidden_state)
            assert last_hidden_state.ndim == 3, last_hidden_state.shape
            assert outputs.logits.ndim == 3, outputs.logits.shape

        logits = outputs.logits[0]
        residue_count = int(input_attention_mask.sum().item()) - 1
        logits = logits[:residue_count]

        invalid_ids = [Settings.TRANSFORMER_TRG_PAD_IDX, SOS_token, EOS_token, OOV_token]
        logits[:, invalid_ids] = -torch.inf
        predicted_ids = logits.argmax(dim=-1)

        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, k=keep_top, dim=-1)
            topk_probs = topk_probs.cpu().numpy()
            topk_ids = topk_ids.cpu().numpy()
            return [i for i in predicted_ids.view(-1).tolist()], list(zip(topk_probs, topk_ids))
        else:
            return [i for i in predicted_ids.view(-1).tolist()]
