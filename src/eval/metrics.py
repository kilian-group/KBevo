""" metrics for evaluating the quality of the predictions """
import re
import string
from collections import Counter

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def exact_match_relaxed(prediction: str, gold_answers: list) -> float:
    """
    MQuAKE-specific EM: 1 if any gold answer appears in the FIRST 5 WORDS of prediction.
    Uses normalized string containment.
    """
    pred_norm = normalize_answer(prediction)
    pred_tokens = pred_norm.split()
    first_five = " ".join(pred_tokens[:5])

    for ans in gold_answers:
        ans_norm = normalize_answer(str(ans))
        if ans_norm and ans_norm in first_five:
            return 1.0
    return 0.0


def mquake_f1_score(prediction: str, gold_answers: list) -> tuple:
    """
    MQuAKE-specific F1: max F1 across all gold answers.
    Returns (f1, precision, recall) for the best matching gold answer.
    """
    best_f1 = 0.0
    best_prec = 0.0
    best_recall = 0.0

    for ans in gold_answers:
        f1, prec, recall = f1_score(prediction, str(ans))
        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_recall = recall

    return best_f1, best_prec, best_recall
