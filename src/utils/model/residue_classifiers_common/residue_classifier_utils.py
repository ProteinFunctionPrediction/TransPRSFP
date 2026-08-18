from abc import ABC, abstractmethod
from utils.model.model_utils import ModelUtils
from utils.utils import Utils
from universal.settings.settings import Settings
from universal.access.universal_access import UniversalAccess

import os
import pickle
import torch
import numpy as np

class ResidueClassifierUtils(ModelUtils, ABC):
    def __init__(self, device) -> None:
        super().__init__(device)

    @abstractmethod
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
        pass

    def get_last_hidden_state(self,
                              encoder,
                              input_sequence,
                              input_attention_mask,
                              embeddings=None,
                              embedding_offset=None,
                              embedding_length=None):
        if encoder is not None:
            with torch.no_grad():
                last_hidden_state = encoder(input_ids=input_sequence, attention_mask=input_attention_mask).last_hidden_state
        else:
            att_len = (input_attention_mask == 1).sum()
            last_hidden_state = np.zeros((1, att_len, 1024), dtype=np.float32)
            offset=embedding_offset
            length=embedding_length
            length = min(length, att_len)
            last_hidden_state[0, :length, :] = embeddings[offset : offset + length]
            last_hidden_state = torch.from_numpy(last_hidden_state).to(self.device)

        return last_hidden_state

    def run_prediction(self,
                       batches,
                       encoder,
                       model,
                       caller,
                       pred_SOS_token_id,
                       pred_EOS_token_id,
                       pred_EMPTY_token_id,
                       pred_OOV_token_id,
                       pred_reverse_word_index,
                       true_SOS_token_id,
                       true_EOS_token_id,
                       true_EMPTY_token_id,
                       true_OOV_token_id,
                       true_reverse_word_index,
                       prefix="",
                       compute_go_based_metrics=False,
                       save_go_based_metrics=True,
                       go_based_metrics_filepath="",
                       go_based_metrics_reg2go_filepath="",
                       compute_metrics=False,
                       dataset_like_region2go_path="",
                       return_probs=False,
                       keep_top=10,
                       embeddings=None,
                       embedding_offset_lookup_table=None):

        if dataset_like_region2go_path != "" and os.path.isfile(dataset_like_region2go_path):
            with open(dataset_like_region2go_path, "rb") as f:
                dataset_like_region2go = pickle.load(f)
        else:
            dataset_like_region2go = None

        if compute_metrics:
            assert true_SOS_token_id is not None
            assert true_EOS_token_id is not None
            assert true_EMPTY_token_id is not None

        probs = None
        if return_probs:
            probs = []

        # reverse_word_index maps ids to strings
        total_go_term_count = len(pred_reverse_word_index) - 1
        dataloader = batches
        predictions = []

        if compute_go_based_metrics:
            go_based_metrics = Utils.build_go_based_metrics_map_from_reverse_go_term_indices(pred_reverse_word_index,
                                                                                             true_reverse_word_index)

            go_based_metrics_reg2go = Utils.build_go_based_metrics_map_from_reverse_go_term_indices(
                pred_reverse_word_index,
                true_reverse_word_index,
                dataset_like_region2go=dataset_like_region2go)
        total_missing_sample_count = 0
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

        if dataset_like_region2go is not None:
            running_metrics[prefix + "precision_reg2go"] = 0.0
            running_metrics[prefix + "recall_reg2go"] = 0.0
            running_metrics[prefix + "accuracy_reg2go"] = 0.0
            running_metrics[prefix + "f1_reg2go"] = 0.0
            running_metrics[prefix + "token_based_precision_reg2go"] = 0.0
            running_metrics[prefix + "token_based_recall_reg2go"] = 0.0
            running_metrics[prefix + "token_based_accuracy_reg2go"] = 0.0
            running_metrics[prefix + "token_based_f1_reg2go"] = 0.0

        with torch.no_grad():
            for idx in range(len(dataloader)):
                # X, y = torch.tensor(X, dtype=torch.long, device=self.device), torch.tensor(y, dtype=torch.long, device=self.device)

                count = idx + 1

                sample_X = torch.tensor(dataloader[idx]["prot_input_ids"], dtype=torch.long,
                                        device=self.device).unsqueeze(0)
                sample_X_attention_mask = torch.tensor(dataloader[idx]["prot_attention_mask"], dtype=torch.long,
                                                       device=self.device).unsqueeze(0)

                sample_y = torch.tensor(dataloader[idx]["go_input_ids"], dtype=torch.long, device=self.device)
                sample_y = sample_y[sample_y != Settings.TRANSFORMER_TRG_PAD_IDX]

                if len(sample_y) > 0 and sample_y[0] == true_SOS_token_id:
                    sample_y = sample_y[1:]

                if len(sample_y) > 0 and sample_y[-1] == true_EOS_token_id:
                    sample_y = sample_y[:-1]

                predict_params = {'encoder': encoder,
                                  'model': model,
                                  'input_sequence': sample_X,
                                  'input_attention_mask': sample_X_attention_mask,
                                  'SOS_token': pred_SOS_token_id,
                                  'EOS_token': pred_EOS_token_id,
                                  'EMPTY_token': pred_EMPTY_token_id,
                                  'OOV_token': pred_OOV_token_id,
                                  'max_length': (sample_X_attention_mask == 1).sum().detach().cpu().numpy() - 1,
                                  'return_probs': return_probs,
                                  'keep_top': keep_top}

                if embeddings is not None and embedding_offset_lookup_table is not None:
                    predict_params['embeddings'] = embeddings
                    predict_params['embedding_offset'] = dataloader[idx]["embedding_offset"]
                    predict_params['embedding_length'] = dataloader[idx]["embedding_length"]

                if return_probs:
                    prediction, sample_probs = self.predict(**predict_params)
                    probs.append(sample_probs)
                else:
                    prediction = self.predict(**predict_params)

                predictions.append(prediction)

                if compute_metrics:
                    assert len(prediction) == len(sample_y)

                    unique_prediction = list(np.unique(np.asarray(prediction)))

                    sample_y_str_list = Utils.convert_idx_list_to_strings(sample_y, true_reverse_word_index)
                    unique_prediction_str_list = Utils.convert_idx_list_to_strings(unique_prediction,
                                                                                   pred_reverse_word_index)
                    prediction_str_list = Utils.convert_idx_list_to_strings(prediction[:len(sample_y)],
                                                                            pred_reverse_word_index)

                    fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(sample_y_str_list,
                                                           unique_prediction_str_list,
                                                           total_go_term_count,
                                                           true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                           pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())

                    precision_score, recall_score, accuracy_score = Utils.precision(tp, fp), Utils.recall(tp,
                                                                                                          fn), Utils.accuracy(
                        fp, tp, fn, tn)
                    f1_score = Utils.f1(precision_score, recall_score)

                    if dataset_like_region2go is not None:
                        total_go_term_count_from_reg2go = Utils.compute_total_go_term_count_from_reg2go(
                            dataset_like_region2go)
                        if "prot_sequence" in dataloader[idx]:
                            protein_sequence_with_spaces = dataloader[idx]["prot_sequence"]
                        else:
                            protein_sequence_with_spaces = caller.get_protein_sequence_with_spaces(idx)
                        if protein_sequence_with_spaces not in dataset_like_region2go:
                            go_map = None
                            total_missing_sample_count += 1
                        else:
                            go_map = dataset_like_region2go[protein_sequence_with_spaces]
                        if go_map is not None:
                            fp_a, tp_a, fn_a, tn_a = Utils.get_fp_tp_fn_tn(
                                Utils.get_unique_go_terms_from_go_map(go_map),
                                unique_prediction_str_list,
                                total_go_term_count_from_reg2go,
                                true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                            precision_score_a, recall_score_a, accuracy_score_a = Utils.precision(tp_a,
                                                                                                  fp_a), Utils.recall(
                                tp_a, fn_a), Utils.accuracy(fp_a, tp_a, fn_a, tn_a)
                            f1_score_a = Utils.f1(precision_score_a, recall_score_a)
                        #                        fp_a, tp_a, fn_a, tn_a = Utils.get_fp_tp_fn_tn_reg2go_analysis(go_map, unique_prediction_str_list, total_go_term_count_from_reg2go)

                        if compute_go_based_metrics and go_map is not None:
                            Utils.update_go_based_metrics_reg2go_map(go_map, prediction_str_list,
                                                                     go_based_metrics_reg2go)

                    match_ratio_score = Utils.match_ratio(sample_y_str_list, prediction_str_list)
                    match_ratio_nonempty_score = Utils.match_ratio_nonempty(sample_y_str_list, prediction_str_list,
                                                                            true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())

                    # _t: abbreviation for _token
                    fp_t, tp_t, fn_t, tn_t = Utils.get_fp_tp_fn_tn_token_based(sample_y_str_list,
                                                                               prediction_str_list,
                                                                               total_go_term_count,
                                                                               true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                                               pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                    precision_score_t, recall_score_t, accuracy_score_t = Utils.precision(tp_t, fp_t), Utils.recall(
                        tp_t, fn_t), \
                        Utils.accuracy(fp_t, tp_t, fn_t, tn_t)
                    f1_score_t = Utils.f1(precision_score_t, recall_score_t)

                    if dataset_like_region2go is not None and go_map is not None:
                        fp_t_a, tp_t_a, fn_t_a, tn_t_a = Utils.get_fp_tp_fn_tn_token_based_reg2go_analysis(go_map,
                                                                                                           prediction_str_list,
                                                                                                           total_go_term_count_from_reg2go,
                                                                                                           true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                                                                           pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                        precision_score_t_a, recall_score_t_a, accuracy_score_t_a = Utils.precision(tp_t_a,
                                                                                                    fp_t_a), Utils.recall(
                            tp_t_a, fn_t_a), \
                            Utils.accuracy(fp_t_a, tp_t_a, fn_t_a, tn_t_a)
                        f1_score_t_a = Utils.f1(precision_score_t_a, recall_score_t_a)

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

                    if dataset_like_region2go is not None and go_map is not None:
                        running_metrics[prefix + "precision_reg2go"] += precision_score_a
                        running_metrics[prefix + "recall_reg2go"] += recall_score_a
                        running_metrics[prefix + "accuracy_reg2go"] += accuracy_score_a
                        running_metrics[prefix + "f1_reg2go"] += f1_score_a

                        running_metrics[prefix + "token_based_precision_reg2go"] += precision_score_t_a
                        running_metrics[prefix + "token_based_recall_reg2go"] += recall_score_t_a
                        running_metrics[prefix + "token_based_accuracy_reg2go"] += accuracy_score_t_a
                        running_metrics[prefix + "token_based_f1_reg2go"] += f1_score_t_a

                    total_sample_count += 1

                    sample_scores = {"count": total_sample_count,
                                     "precision": precision_score,
                                     "recall": recall_score,
                                     "accuracy": accuracy_score,
                                     "f1": f1_score,
                                     "match_ratio": match_ratio_score,
                                     "match_ratio_nonempty": match_ratio_nonempty_score,
                                     "token_based_precision": precision_score_t,
                                     "token_based_recall": recall_score_t,
                                     "token_based_accuracy": accuracy_score_t,
                                     "token_based_f1": f1_score_t}

                    if dataset_like_region2go is not None and go_map is not None:
                        sample_scores["precision_reg2go"] = precision_score_a
                        sample_scores["recall_reg2go"] = recall_score_a
                        sample_scores["accuracy_reg2go"] = accuracy_score_a
                        sample_scores["f1_reg2go"] = f1_score_a
                        sample_scores["token_based_precision_reg2go"] = precision_score_t_a
                        sample_scores["token_based_recall_reg2go"] = recall_score_t_a
                        sample_scores["token_based_accuracy_reg2go"] = accuracy_score_t_a
                        sample_scores["token_based_f1_reg2go"] = f1_score_t_a
                    print(sample_scores)

                if caller is not None:
                    prediction_str = caller.post_process_prediction_as_str(
                        caller.post_process_prediction(model, prediction, pred_EMPTY_token_id, pred_reverse_word_index))
                    if "prot_sequence" in dataloader[idx]:
                        protein_sequence = dataloader[idx]["prot_sequence"].replace(" ", "")
                        UniversalAccess.output.write(f"{protein_sequence}: {prediction_str}")
                    else:
                        caller.notify(count - 1, prediction_str)
        average_metrics = {}
        if compute_metrics:
            for key, value in running_metrics.items():
                if "_reg2go" in key:
                    average_metrics[key] = value / (total_sample_count - total_missing_sample_count)
                else:
                    average_metrics[key] = value / total_sample_count

        if compute_go_based_metrics and save_go_based_metrics and go_based_metrics_filepath != "":
            if os.path.exists(go_based_metrics_filepath):
                print(f"{go_based_metrics_filepath} already exists! Won't save GO term-based metrics:")
                print(go_based_metrics)
            else:
                try:
                    with open(go_based_metrics_filepath, "wb") as go_based_metrics_file:
                        pickle.dump(go_based_metrics, go_based_metrics_file)
                except Exception as e:
                    print(
                        f"An error occurred while saving GO term-based metrics to {os.path.abspath(go_based_metrics_filepath)}: {str(e)}")
                else:
                    print(f"GO term-based metrics have been saved to {os.path.abspath(go_based_metrics_filepath)}")
        elif compute_go_based_metrics:
            print("GO term-based metrics:")
            print(go_based_metrics)

        if dataset_like_region2go is not None and compute_go_based_metrics and save_go_based_metrics and go_based_metrics_reg2go_filepath != "":
            if os.path.exists(go_based_metrics_reg2go_filepath):
                print(f"{go_based_metrics_reg2go_filepath} already exists! Won't save GO term-based reg2go metrics:")
                print(go_based_metrics)
            else:
                try:
                    with open(go_based_metrics_reg2go_filepath, "wb") as go_based_metrics_file:
                        pickle.dump(go_based_metrics_reg2go, go_based_metrics_file)
                except Exception as e:
                    print(
                        f"An error occurred while saving GO term-based reg2go metrics to {os.path.abspath(go_based_metrics_reg2go_filepath)}: {str(e)}")
                else:
                    print(
                        f"GO term-based reg2go metrics have been saved to {os.path.abspath(go_based_metrics_reg2go_filepath)}")
        elif dataset_like_region2go is not None and compute_go_based_metrics:
            print("GO term-based reg2go metrics:")
            print(go_based_metrics_reg2go)

        if return_probs:
            return average_metrics, predictions, probs
        else:
            return average_metrics, predictions