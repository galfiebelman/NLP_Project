import json
import random
from tqdm import tqdm

import pandas as pd
import torch
import numpy as np
from accelerate import Accelerator
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from allennlp.training.metrics.f1_measure import F1Measure
from typing import Optional

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce_sum

PREPOSITIONS = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                'inside', 'outside', 'into', 'around']

prep_dict = {prep: idx for idx, prep in enumerate(PREPOSITIONS)}


class MCF1Measure(Metric):
    """
    An adaptation of the allennlp.training.metrics.fbeta_measure.FBetaMeasure class
    to be used with multi-class, but allowing to ignore one of the labels as part
    of the calculation.

    Compute precision, recall, F-measure and support for each class.

    The precision is the ratio `tp / (tp + fp)` where `tp` is the number of
    true positives and `fp` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio `tp / (tp + fn)` where `tp` is the number of
    true positives and `fn` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    If we have precision and recall, the F-beta score is simply:
    `F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)`

    The F-beta score weights recall more than precision by a factor of
    `beta`. `beta == 1.0` means recall and precision are equally important.

    The support is the number of occurrences of each class in `y_true`.

    # Parameters

    beta : `float`, optional (default = `1.0`)
        The strength of recall versus precision in the F-score.

    """

    def __init__(self, beta: float = 1.0) -> None:
        if beta <= 0:
            raise ConfigurationError("`beta` should be >0 in the F-beta score.")
        self._beta = beta

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: int = -1
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: int = -1
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: int = -1
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: int = -1

    @overrides
    def __call__(
            self,
            predictions: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum == -1:
            self._true_positive_sum = 0
            self._true_sum = 0
            self._pred_sum = 0
            self._total_sum = 0

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = predictions.sum(dim=-1) != 0

        links = predictions != 0

        true_positives = (gold_labels == predictions) & links & mask & pred_mask

        true_positive_sum = torch.sum(true_positives).item()

        preds = (predictions != 0) & mask & pred_mask
        pred_sum = preds.sum().item()

        gold_labels = (gold_labels != 0) & mask
        true_sum = gold_labels.sum().item()

        self._true_positive_sum += dist_reduce_sum(true_positive_sum)
        self._pred_sum += dist_reduce_sum(pred_sum)
        self._true_sum += dist_reduce_sum(true_sum)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        precisions : `List[float]`
        recalls : `List[float]`
        f1-measures : `List[float]`

        !!! Note
            If `self.average` is not `None`, you will get `float` instead of `List[float]`.
        """
        if self._true_positive_sum == -1:
            raise RuntimeError("You never call this metric before.")

        else:
            tp_sum = self._true_positive_sum
            pred_sum = self._pred_sum
            true_sum = self._true_sum

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        if pred_sum == 0:
            precision = 0
        else:
            precision = tp_sum / pred_sum
        if true_sum == 0:
            recall = 0
        else:
            recall = tp_sum / true_sum
        if precision == 0 or recall == 0:
            fscore = 0
        else:
            fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)

        if reset:
            self.reset()

        return {"precision": precision, "recall": recall, "fscore": fscore}

    @overrides
    def reset(self) -> None:
        self._true_positive_sum = -1
        self._pred_sum = -1
        self._true_sum = -1


def encode_data(tokenizer, questions, passages):
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(question, passage, pad_to_max_length=True,
                                             max_length=256,
                                             truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)

    return np.array(input_ids), np.array(attention_masks)


def create_questions_answers(texts, np_relations, nps):
    out_texts = []
    out_questions = []
    out_answers = []

    # for text, relations, phrases in zip(texts, np_relations, nps):
    #     for i, phrase_1 in enumerate(phrases):
    #         answer_df = pd.DataFrame.from_records(relations)
    #         anchor = phrases[phrase_1]['text']
    #         for j, phrase_2 in enumerate(phrases):
    #             if i == j:
    #                 continue
    #             comp = phrases[phrase_2]['text']
    #             question = "what is the noun phrase relation between {} and {}".format(anchor, comp)
    #             cur_answer_df = answer_df[answer_df['anchor'] == phrase_1]
    #             if len(cur_answer_df) > 0:
    #                 cur_answer_df = cur_answer_df[cur_answer_df['complement'] == phrase_2]
    #             if len(cur_answer_df) == 0:
    #                 answer = prep_dict["no-relation"]
    #             else:
    #                 answer = prep_dict[cur_answer_df['preposition'].values[0]]
    #             out_texts.append(text)
    #             out_questions.append(question.lower())
    #             out_answers.append(answer)
    #
    # return out_texts, out_questions, out_answers

    for text, relations, phrases in zip(texts, np_relations, nps):
        relations_df = pd.DataFrame.from_records(relations)
        for i, phrase_1 in enumerate(phrases):
            anchor = phrases[phrase_1]['text']
            cur_relations_df = relations_df[relations_df['anchor'] == phrase_1]
            if len(cur_relations_df) == 0:
                continue
            comps = list(cur_relations_df['complement'].values)
            no_comps = np.random.permutation(np.array([nph for nph in list(phrases.keys()) if nph not in comps]))
            max_len = len(comps) if len(comps) < len(no_comps) else len(no_comps)
            preps = list(cur_relations_df['preposition'].values)
            for i, comp_phrase in enumerate(comps):
                comp = phrases[comp_phrase]['text']
                question = "What is the noun phrase relation between {} and {}?".format(anchor, comp)
                out_texts.append(text)
                out_questions.append(question)
                out_answers.append(prep_dict[preps[i]])
                if i < max_len:
                    b_comp = no_comps[i]
                    b_comp = phrases[b_comp]['text']
                    b_question = "What is the noun phrase relation between {} and {}?".format(anchor, b_comp)
                    out_texts.append(text)
                    out_questions.append(b_question)
                    out_answers.append(0)

    return out_texts, out_questions, out_answers


def create_and_save_trainable_data_loader(tokenizer, data_df, batch_size):
    texts = data_df.text.values
    nps = data_df.nps.values
    np_relations = data_df.np_relations.values
    # corefs = data_df.coref.values
    texts, questions, answers = create_questions_answers(texts, np_relations, nps)
    print("encoding")
    input_ids, attention_masks = encode_data(tokenizer, questions, texts)
    features = (input_ids, attention_masks, answers)
    features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in features]
    dataset = TensorDataset(*features_tensors)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    # torch.save(dataloader, save_path)
    return dataloader


def train_model(model, epochs, grad_acc_steps, optimizer, train_loader, dev_loader, acc):
    train_loss_values = []
    dev_acc_values = []
    best_acc = 0
    cnt = 0
    best_model = model
    for _ in tqdm(range(epochs), desc="Epoch"):

        # Training
        epoch_train_loss = 0  # Cumulative loss
        num_samples = 0
        model.train()
        model.zero_grad()
        # curr_paths = np.random.permutation(np.array(train_loader_paths))
        # for path in curr_paths:
        # train_loader = torch.load(path)
        # num_samples += len(train_loader)
        for step, batch in enumerate(train_loader):

            input_ids = batch[0]  # .to(device)
            attention_masks = batch[1]  # .to(device)
            labels = batch[2]  # .to(device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)

            loss = outputs[0]
            loss = loss / grad_acc_steps
            epoch_train_loss += loss.item()

            # loss.backward()
            acc.backward(loss)

            if (step + 1) % grad_acc_steps == 0:  # Gradient accumulation is over
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clipping gradients
                optimizer.step()
                model.zero_grad()

            # epoch_train_loss += epoch_train_loss
            # train_loader = None
        epoch_train_loss = epoch_train_loss / len(train_loader)
        print("train loss:{}".format(epoch_train_loss))
        train_loss_values.append(epoch_train_loss)
        # train_loader = None
        # Evaluation
        epoch_dev_accuracy = 0  # Cumulative accuracy
        model.eval()

        # dev_loader = torch.load(dev_loader_path)
        for batch in dev_loader:
            input_ids = batch[0]  # .to(device)
            attention_masks = batch[1]  # .to(device)
            labels = batch[2]

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()

            predictions = np.argmax(logits, axis=1).flatten()
            labels = labels.detach().cpu().numpy().flatten()

            epoch_dev_accuracy += np.sum(predictions == labels) / len(labels)

        epoch_dev_accuracy = epoch_dev_accuracy / len(dev_loader)
        # dev_loader = None
        print("dev acc:{}".format(epoch_dev_accuracy))
        dev_acc_values.append(epoch_dev_accuracy)
        if epoch_dev_accuracy > best_acc:
            best_acc = epoch_dev_accuracy
            print("saved")
            # model.save_pretrained(model_path)
            best_model = model
            cnt = 0
        else:
            cnt += 1
            if cnt == epochs // 3:
                return train_loss_values, dev_acc_values, best_model

    return train_loss_values, dev_acc_values, best_model


def evaluate(model, output_file, tokenizer, acc):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(PREPOSITIONS))
    # model.to(device)
    model.eval()
    test_df = pd.read_json("dev.jsonl", lines=True, orient='records').head(1)
    texts = test_df.text.values
    nps = test_df.nps.values
    dicts = []
    for text, phrases in zip(texts, nps):
        cur_dict = {"predicted_prepositions": []}
        for i, phrase_1 in enumerate(phrases):
            anchor = phrases[phrase_1]['text']
            for j, phrase_2 in enumerate(phrases):
                if i == j:
                    cur_dict["predicted_prepositions"].append(0)
                    continue
                comp = phrases[phrase_2]['text']
                question = "What is the noun phrase relation between {} and {}?".format(anchor, comp)
                sequence = tokenizer.encode_plus(question, text, return_tensors="pt")['input_ids'].to(acc.device)
                with torch.no_grad():
                    out = model(sequence)[0]
                probabilities = torch.softmax(out, dim=1).detach().cpu().tolist()[0]
                pred = np.argmax(np.array(probabilities))
                cur_dict["predicted_prepositions"].append(int(pred))
        dicts.append(cur_dict)

    with open(output_file, 'w') as out_file:
        for d in dicts:
            out_file.write(json.dumps(d))
            out_file.write("\n")
    return out_file


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
    print(results)
    print('=== Writing results to file ===')
    to_file(results, output_file)

    print('=== Done ===')


if __name__ == "__main__":
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)
    accelerator = Accelerator()
    batch_size = 32
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    all_df = pd.read_json('train.jsonl', lines=True, orient='records').head(2)
    num_dev_texts = 1
    dev_df = all_df.tail(num_dev_texts)
    train_df = all_df.head(len(all_df) - num_dev_texts)
    train_loader = create_and_save_trainable_data_loader(tokenizer, dev_df, batch_size)
    dev_loader = create_and_save_trainable_data_loader(tokenizer, dev_df, batch_size)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                               num_labels=len(PREPOSITIONS))
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    epochs = 1
    grad_acc_steps = 1
    model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)
    train_losses, dev_accs, model = train_model(model, epochs, grad_acc_steps, optimizer, train_loader, dev_loader,
                                                accelerator)
    print(train_losses)
    print(dev_accs)
    output_file = 'eval_predictions.jsonl'
    out_file = evaluate(model, output_file, tokenizer, accelerator)
    eval_file = 'dev_eval.jsonl'
    score_file = 'score.jsonl'
    score(output_file, eval_file, score_file)
