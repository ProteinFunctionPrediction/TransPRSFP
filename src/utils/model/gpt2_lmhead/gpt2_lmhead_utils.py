from utils.model.model_utils import ModelUtils
from utils.dataset.dataset_utils import DatasetUtils
from universal.settings.settings import Settings
from utils.utils import Utils
from io import StringIO
import os
import numpy as np
import torch
import json
import pickle

class GPT2LMHeadUtils(ModelUtils):
    def __init__(self, device) -> None:
        super().__init__(device)
    
    def run_prediction(self,
                       batches,
                       encoder,
                       model,
                       caller,
                       pred_SOS_token_id,
                       pred_EOS_token_id,
                       pred_EMPTY_token_id,
                       pred_reverse_word_index,
                       true_SOS_token_id,
                       true_EOS_token_id,
                       true_EMPTY_token_id,
                       true_reverse_word_index,
                       prefix="",
                       compute_go_based_metrics=False,
                       save_go_based_metrics=True,
                       go_based_metrics_filepath="",
                       compute_metrics=False):
        if compute_metrics:
            assert true_SOS_token_id is not None
            assert true_EOS_token_id is not None
            assert true_EMPTY_token_id is not None
        
        # reverse_word_index maps ids to strings
        total_go_term_count = len(pred_reverse_word_index) - 1
        dataloader = batches
        predictions = []

        if compute_go_based_metrics:
            go_based_metrics = Utils.build_go_based_metrics_map_from_reverse_go_term_indices(pred_reverse_word_index,
                                                                                             true_reverse_word_index)
        
        total_sample_count = 0
        running_metrics = {prefix + "precision": 0.0,
                           prefix + "recall": 0.0,
                           prefix + "accuracy": 0.0,
                           prefix + "f1": 0.0,
                           prefix + "match_ratio": 0.0,
                           prefix + "match_ratio_nonempty": 0.0,
                           prefix + "token_based_precision": 0.0,
                           prefix + "token_based_accuracy": 0.0,
                           prefix + "token_based_recall": 0.0,
                           prefix + "token_based_f1": 0.0}
        
        with torch.no_grad():
            for idx in range(len(dataloader)):
                #X, y = torch.tensor(X, dtype=torch.long, device=self.device), torch.tensor(y, dtype=torch.long, device=self.device)
                
                count = idx + 1
                
                sample_X = torch.tensor(dataloader[idx]["prot_input_ids"], dtype=torch.long, device=self.device).unsqueeze(0)
                sample_X_attention_mask = torch.tensor(dataloader[idx]["prot_attention_mask"], dtype=torch.long, device=self.device).unsqueeze(0)
                
                sample_y = torch.tensor(dataloader[idx]["go_input_ids"], dtype=torch.long, device=self.device)
                sample_y = sample_y[sample_y != Settings.TRANSFORMER_TRG_PAD_IDX]
                
                if len(sample_y) > 0 and sample_y[0] == true_SOS_token_id:
                    sample_y = sample_y[1:]
                
                if len(sample_y) > 0 and sample_y[-1] == true_EOS_token_id:
                    sample_y = sample_y[:-1]

                
                prediction = self.predict(encoder=encoder,
                                          model=model,
                                          input_sequence=sample_X,
                                          input_attention_mask=sample_X_attention_mask,
                                          SOS_token=pred_SOS_token_id,
                                          EOS_token=pred_EOS_token_id,
                                          EMPTY_token=pred_EMPTY_token_id,
                                          max_length=(sample_X_attention_mask == 1).sum().detach().cpu().numpy() - 1)[1:]
                
                predictions.append(prediction)
                
                if compute_metrics:
                    assert len(prediction) == len(sample_y)
                
                    unique_prediction = list(np.unique(np.asarray(prediction)))
                
                    sample_y_str_list = Utils.convert_idx_list_to_strings(sample_y, true_reverse_word_index)
                    unique_prediction_str_list = Utils.convert_idx_list_to_strings(unique_prediction, pred_reverse_word_index)
                    prediction_str_list = Utils.convert_idx_list_to_strings(prediction[:len(sample_y)], pred_reverse_word_index)
                
                    fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(sample_y_str_list,
                                                           unique_prediction_str_list,
                                                           total_go_term_count,
                                                           true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                           pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                    precision_score, recall_score, accuracy_score = Utils.precision(tp, fp), Utils.recall(tp, fn), Utils.accuracy(fp, tp, fn, tn)
                    f1_score = Utils.f1(precision_score, recall_score)

                    match_ratio_score = Utils.match_ratio(sample_y_str_list, prediction_str_list)
                    match_ratio_nonempty_score = Utils.match_ratio_nonempty(sample_y_str_list, prediction_str_list, true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                
                    # _t: abbreviation for _token
                    fp_t, tp_t, fn_t, tn_t = Utils.get_fp_tp_fn_tn_token_based(sample_y_str_list,
                                                                               prediction_str_list,
                                                                               total_go_term_count,
                                                                               true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                                               pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                    precision_score_t, recall_score_t, accuracy_score_t = Utils.precision(tp_t, fp_t), Utils.recall(tp_t, fn_t), \
                                                                        Utils.accuracy(fp_t, tp_t, fn_t, tn_t)
                    f1_score_t = Utils.f1(precision_score_t, recall_score_t)
                
                    if compute_go_based_metrics:
                        Utils.update_go_based_metrics_map(sample_y_str_list,
                                                          prediction_str_list,
                                                          go_based_metrics)
                
                    running_metrics[prefix + "precision"] += precision_score
                    running_metrics[prefix + "recall"] += recall_score
                    running_metrics[prefix + "accuracy"] += accuracy_score
                    running_metrics[prefix + "f1"] += f1_score

                    running_metrics[prefix + "match_ratio"] += match_ratio_score
                    running_metrics[prefix + "match_ratio_nonempty"] += match_ratio_nonempty_score
                    running_metrics[prefix + "token_based_precision"] += precision_score_t
                    running_metrics[prefix + "token_based_recall"] += recall_score_t
                    running_metrics[prefix + "token_based_accuracy"] += accuracy_score_t
                    running_metrics[prefix + "token_based_f1"] += f1_score_t
                    total_sample_count += 1

                    print({"count": total_sample_count,
                           "precision": precision_score,
                           "recall": recall_score,
                           "accuracy": accuracy_score,
                           "f1": f1_score,
                           "match_ratio": match_ratio_score,
                           "match_ratio_nonempty": match_ratio_nonempty_score,
                           "token_based_precision": precision_score_t,
                           "token_based_recall": recall_score_t,
                           "token_based_accuracy": accuracy_score_t,
                           "token_based_f1": f1_score_t})

                if caller is not None:
                    caller.notify(count - 1, self.post_process_prediction_as_str(self.post_process_prediction(model, prediction, pred_EMPTY_token_id, pred_reverse_word_index)))
        
        average_metrics = {}
        if compute_metrics:
            for key, value in running_metrics.items():
                average_metrics[key] = value/total_sample_count
        
        if compute_go_based_metrics and save_go_based_metrics and go_based_metrics_filepath != "":
            if os.path.exists(go_based_metrics_filepath):
                print(f"{go_based_metrics_filepath} already exists! Won't save GO term-based metrics:")
                print(go_based_metrics)
            else:
                try:
                    with open(go_based_metrics_filepath, "wb") as go_based_metrics_file:
                        pickle.dump(go_based_metrics, go_based_metrics_file)
                except Exception as e:
                    print(f"An error occurred while saving GO term-based metrics to {os.path.abspath(go_based_metrics_filepath)}: {str(e)}")
                else:
                    print(f"GO term-based metrics have been saved to {os.path.abspath(go_based_metrics_filepath)}")
        elif compute_go_based_metrics:
            print("GO term-based metrics:")
            print(go_based_metrics)
        
        return average_metrics, predictions

    def predict(self, encoder, model, input_sequence, input_attention_mask, max_length, SOS_token, EOS_token, EMPTY_token):
        model.eval()
        with torch.no_grad():
            last_hidden_state = encoder(input_ids=input_sequence, attention_mask=input_attention_mask).last_hidden_state

        y_input = torch.tensor([SOS_token], dtype=torch.long, device=self.device)
        
        for _ in range(max_length):
            pred = model(input_ids=y_input, encoder_hidden_states=last_hidden_state)
            next_token_logits = pred.logits[-1, :]
            next_token_logits[SOS_token] = -1e20
            next_token_logits[EOS_token] = -1e20
            next_token_logits[Settings.TRANSFORMER_TRG_PAD_IDX] = -1e20
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            y_input = torch.cat((y_input, next_token), dim=-1)
        
        return [i for i in y_input.view(-1).tolist()]
    
    def post_process_prediction(self, model, prediction, empty_token, reverse_word_index):        
        go_term_to_region = dict()
        if len(prediction) == 0:
            return go_term_to_region

        last_different_token = prediction[0]
        region_start_idx = 0
        region_end_idx = region_start_idx
        
        for idx in range(1, len(prediction)):
            current_token = prediction[idx]
            if current_token == last_different_token:
                region_end_idx += 1
            else:
                if last_different_token not in go_term_to_region:
                    go_term_to_region[last_different_token] = list()
                go_term_to_region[last_different_token].append((region_start_idx, region_end_idx))
            
                last_different_token = current_token
                region_start_idx = region_end_idx + 1
                region_end_idx = region_start_idx
        
        if last_different_token not in go_term_to_region:
            go_term_to_region[last_different_token] = list()
        
        go_term_to_region[last_different_token].append((region_start_idx, region_end_idx))
        
        result = dict()
        for key, value in go_term_to_region.items():
            regions = [[i[0] + 1, i[1] + 1] for i in value]
            result[reverse_word_index[key]] = regions
        return result

    def post_process_prediction_as_str(self, post_process_result: dict) -> str:
        string_io = StringIO()
        for key, value in post_process_result.items():
            string_io.write(f"{key}: {json.dumps(value)} | ")
        return string_io.getvalue()
