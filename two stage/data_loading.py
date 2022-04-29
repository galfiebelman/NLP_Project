import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import torch

PREPOSITIONS = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                'inside', 'outside', 'into', 'around']
RELATIONS = ['of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
             'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
             'inside', 'outside', 'into', 'around']
ANSWERS = ['no', 'yes']
prep_dict = {prep: idx for idx, prep in enumerate(RELATIONS)}


def encode_data(tokenizer, questions, passages, max_length):
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True,
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

    for text, relations, phrases in zip(texts, np_relations, nps):
        for i, phrase_1 in enumerate(phrases):
            answer_df = pd.DataFrame.from_records(relations)
            anchor = phrases[phrase_1]['text']
            for j, phrase_2 in enumerate(phrases):
                if i == j:
                    continue
                comp = phrases[phrase_2]['text']
                question = "what is the relation between {} and {}?".format(anchor, comp)
                cur_answer_df = answer_df[answer_df['anchor'] == phrase_1]
                if len(cur_answer_df) > 0:
                    cur_answer_df = cur_answer_df[cur_answer_df['complement'] == phrase_2]
                if len(cur_answer_df) != 0:
                    answer = prep_dict[cur_answer_df['preposition'].values[0]]
                    out_texts.append(text)
                    out_questions.append(question.lower())
                    out_answers.append(answer)

    return out_texts, out_questions, out_answers


def create_binary_questions_answers(texts, np_relations, nps):
    out_texts = []
    out_questions = []
    out_answers = []

    for text, relations, phrases in zip(texts, np_relations, nps):
        for i, phrase_1 in enumerate(phrases):
            answer_df = pd.DataFrame.from_records(relations)
            anchor = phrases[phrase_1]['text']
            for j, phrase_2 in enumerate(phrases):
                if i == j:
                    continue
                comp = phrases[phrase_2]['text']
                question = "Is there a relation between {} and {}?".format(anchor, comp)
                cur_answer_df = answer_df[answer_df['anchor'] == phrase_1]
                if len(cur_answer_df) > 0:
                    cur_answer_df = cur_answer_df[cur_answer_df['complement'] == phrase_2]
                if len(cur_answer_df) != 0:
                    answer = 0
                else:
                    answer = 1
                out_texts.append(text)
                out_questions.append(question.lower())
                out_answers.append(answer)

    return out_texts, out_questions, out_answers

# def count_questions_answers(texts, np_relations, nps):
#     out_texts = []
#     out_questions = []
#     out_answers = []
#
#     num_relation = 0
#     num_no_relation = 0
#     for text, relations, phrases in zip(texts, np_relations, nps):
#         for i, phrase_1 in enumerate(phrases):
#             answer_df = pd.DataFrame.from_records(relations)
#             anchor = phrases[phrase_1]['text']
#             for j, phrase_2 in enumerate(phrases):
#                 if i == j:
#                     continue
#                 comp = phrases[phrase_2]['text']
#                 cur_answer_df = answer_df[answer_df['anchor'] == phrase_1]
#                 if len(cur_answer_df) > 0:
#                     cur_answer_df = cur_answer_df[
#                         cur_answer_df['complement'] == phrase_2]
#                 if len(cur_answer_df) == 0:
#                     num_no_relation = num_no_relation + 1
#                 else:
#                     num_relation = num_relation + 1
#     print("relation" + str(num_relation))
#     print("no relation" + str(num_no_relation))
#     return out_texts, out_questions, out_answers

# def create_coref_questions_answers(texts, corefs, nps):
#     out_texts = []
#     out_questions = []
#     out_answers = []
#
#     for text, clusters, phrases in zip(texts, corefs, nps):
#         for i, phrase_1 in enumerate(phrases):
#             anchor = phrases[phrase_1]['text']
#             for j, phrase_2 in enumerate(phrases):
#                 if i == j:
#                     continue
#                 comp = phrases[phrase_2]['text']
#                 question = "is {} a coreference of {}".format(anchor, comp)
#                 answer = 0
#                 for clust in clusters:
#                     members = clust['members']
#                     if len(members) == 0:
#                         continue
#                     if phrase_1 in members and phrase_2 in members:
#                         answer = 1
#                         break
#                 out_texts.append(text)
#                 out_questions.append(question.lower())
#                 out_answers.append(answer)
#
#     return out_texts, out_questions, out_answers


# def create_trainable_data_loader(tokenizer, data_df, max_seq_length, batch_size):
#     texts = data_df.text.values
#     nps = data_df.nps.values
#     np_relations = data_df.np_relations.values
#     # corefs = data_df.coref.values
#     texts, questions, answers = create_questions_answers(texts, np_relations, nps)
#
#     input_ids, attention_masks = encode_data(tokenizer, questions, texts, max_seq_length)
#     features = (input_ids, attention_masks, answers)
#     features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in features]
#     dataset = TensorDataset(*features_tensors)
#     sampler = RandomSampler(dataset)
#     dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
#     return dataloader

def create_trainable_dataset(tokenizer, data_df, max_seq_length, binary=False):
    texts = data_df.text.values
    nps = data_df.nps.values
    np_relations = data_df.np_relations.values
    # corefs = data_df.coref.values
    if binary:
        texts, questions, answers = create_binary_questions_answers(texts,
                                                                    np_relations,
                                                                    nps)
    else:
        texts, questions, answers = create_questions_answers(texts,
                                                             np_relations, nps)

    input_ids, attention_masks = encode_data(tokenizer, questions, texts, max_seq_length)
    features = (input_ids, attention_masks, answers)
    features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in features]
    dataset = TensorDataset(*features_tensors)
    return dataset
