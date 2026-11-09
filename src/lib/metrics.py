import numpy as np
from collections import OrderedDict

def precision(tp, fp):
    if tp + fp == 0:
        return float('nan')
    return tp / (tp + fp)


def recall(tp, fn):
    if tp + fn == 0:
        return float('nan')
    return tp / (tp + fn)


def accuracy(fp, tp, fn, tn):
    if fp + tp + fn + tn == 0:
        return float('nan')
    return (tp + tn) / (fp + tp + fn + tn)

def f1(tp, fp, fn):
    if tp + fp + fn == 0:
        return float('nan')
    return (2 * tp)/((2 * tp) + fp + fn)

def f1_prec_recall(precision, recall):
    if np.isnan(precision) and np.isnan(recall):
        return float('nan')

    if np.isnan(precision) or np.isnan(recall):
        return 0

    if precision + recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))

def compute_metrics(go):
    result = {}
    for token, metrics in go.items():
        tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        f = f1(tp, fp, fn)
        acc = accuracy(fp, tp, fn, tn)
        result[token.upper()] = {'precision': prec, 'recall': rec, 'f1': f, 'accuracy': acc}
    
    metric_based_result = {}
    for metric_name in ['precision', 'recall', 'f1', 'accuracy']:
        metric_res = {}
        for token, token_metrics in result.items():
            metric_res[token] = token_metrics[metric_name]
        metric_based_result[metric_name] = OrderedDict(sorted(metric_res.items(), key=lambda x: x[1], reverse=True))
    
    return result, metric_based_result

def compute_total_tp_fp_tn_fn(go, use_go_terms_only=None, exclude_go_terms=None):
    if exclude_go_terms is None:
        exclude_go_terms = set(['<empty>', '<EMPTY>', '<oov>', '<OOV>'])

    if use_go_terms_only is None:
        use_go_terms_only = set()

    tp_t = 0
    fp_t = 0
    tn_t = 0
    fn_t = 0
    for token, metrics in go.items():
        if token in exclude_go_terms or (len(use_go_terms_only) > 0 and token not in use_go_terms_only):
            continue
        tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
        tp_t += tp
        fp_t += fp
        tn_t += tn
        fn_t += fn

    return tp_t, fp_t, tn_t, fn_t

def calculate_average_values(metrics, use_go_terms_only=None, exclude_go_terms=None):
    if exclude_go_terms is None:
        exclude_go_terms = set(['<empty>', '<EMPTY>', '<oov>', '<OOV>'])
    
    if use_go_terms_only is None:
        use_go_terms_only = set()
    
    metric_names = ["precision", "recall", "f1", "accuracy"]
    
    result = {}
    
    for metric_name in metric_names:
        s = 0
        count = 0
        for go_term, value in metrics[metric_name].items():
            if np.isnan(value) or go_term in exclude_go_terms or (len(use_go_terms_only) > 0 and go_term not in use_go_terms_only):
                continue
            s += value
            count += 1
        if count > 0:
            result[metric_name] = s/count
        else:
            result[metric_name] = 0
    return result
