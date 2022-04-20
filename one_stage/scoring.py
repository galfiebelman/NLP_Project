import argparse
import json
import random

import torch
import numpy as np

from allennlp.training.metrics.f1_measure import F1Measure
from mcf1_measure import MCF1Measure

preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']

preposition_dict = {k: v for v, k in enumerate(preposition_list)}


def load_data(in_f):
    with open(in_f, 'r') as f:
        data = f.readlines()

    data = [json.loads(x.strip()) for x in data]
    return data


def to_file(result, out_f):
    with open(out_f, 'w') as f:
        json.dump(result, f)


def get_predicted_labels(doc):
    """
    Assuming n^2 links, where n is the number of NPs in the document
    Each location corresponds to the natural order of the NPs combination:
     i*n + j, when i the row and j is the column
    In the gold labels, the diagonal (self-links) will have the -1 value
     and these won't be part of the calculation
    If there's no link, assuming a 0-value, any other value means there's a link,
     and the number corresponds to the preposition id.
    The mapping of prepositions and ids is based on the `preposition_list` variable

    """
    prepositions = np.array(doc['predicted_prepositions'])
    relation_exists = (prepositions != 0) * 1
    return relation_exists, prepositions


def get_gold_labels(doc):
    relation_exists = np.array(doc['relations'])
    prepositions = np.array(doc['prepositions'], dtype=object)
    return relation_exists, prepositions


def adapt_prep_labels_to_predictions(extended_prepositions_labels, preposition_predictions):
    updated_labels = []
    for prep_p, prep_extend in zip(preposition_predictions, extended_prepositions_labels):
        if prep_p in prep_extend:
            updated_labels.append(prep_p)
        else:
            updated_labels.append(prep_extend[-1])
    return np.array(updated_labels)


def evaluate_documents(gold_docs, pred_docs):
    labeled_f1 = MCF1Measure()
    unlabeled_f1 = F1Measure(positive_label=1)

    for gold_doc, pred_doc in zip(gold_docs, pred_docs):
        gold_relation, gold_preposition = get_gold_labels(gold_doc)
        predicted_link, predicted_preposition = get_predicted_labels(pred_doc)

        adapted_preposition_labels = adapt_prep_labels_to_predictions(gold_preposition, predicted_preposition)

        unlabeled_f1(
            torch.tensor(np.concatenate([predicted_link.reshape(-1, 1) == 0,
                                         predicted_link.reshape(-1, 1) == 1],
                                        axis=1))
                .reshape(1, -1, 2),
            torch.tensor(gold_relation.reshape(1, -1)),
            torch.tensor(gold_relation != -1).reshape(1, -1)
        )

        labeled_f1(torch.tensor(predicted_preposition),
                   torch.tensor(adapted_preposition_labels),
                   (gold_relation != -1))

    link_f1_score = unlabeled_f1.get_metric(reset=True)
    f1_score = labeled_f1.get_metric(reset=True)

    scores = {
        'labeled_p': f1_score['precision'],
        'labeled_r': f1_score['recall'],
        'labeled_f1': f1_score['fscore'],
        'unlabeled_p': link_f1_score['precision'],
        'unlabeled_r': link_f1_score['recall'],
        'unlabeled_f1': link_f1_score['f1'],
    }
    return scores


def score(predictions_file, gold_file, output_file):
    print('=== Reading answers and predictions files ===')
    document_answers = load_data(gold_file)
    document_preds = load_data(predictions_file)

    print('=== Evaluating... ===')
    results = evaluate_documents(document_answers, document_preds)

    print('=== Writing results to file ===')
    to_file(results, output_file)

    print('=== Done ===')
