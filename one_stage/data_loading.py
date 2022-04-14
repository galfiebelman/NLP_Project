import random

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import torch
from transformers import AutoTokenizer, AlbertTokenizer

PREPOSITIONS = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                'inside', 'outside', 'into', 'around']
prep_dict = {prep: idx for idx, prep in enumerate(PREPOSITIONS)}


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


def create_and_save_trainable_data_loader(tokenizer, data_df, batch_size, save_path):
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
    torch.save(dataloader, save_path)


if __name__ == "__main__":
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)
    model_type = 'distilbert'
    batch_size = 32
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loaders_dir = model_type + '_loaders/'
    all_df = pd.read_json('train.jsonl', lines=True, orient='records')
    num_loaders = 32
    num_dev_texts = 400
    dev_df = all_df.tail(num_dev_texts)
    create_and_save_trainable_data_loader(tokenizer, dev_df, batch_size, loaders_dir + "dev_loader.pth")
    dev_df = None
    all_df = all_df.head(len(all_df) - num_dev_texts)
    for i in range(num_loaders):
        if i == num_loaders - 1:
            train_df = all_df.iloc[(len(all_df) // num_loaders) * i:, :]
            create_and_save_trainable_data_loader(tokenizer, train_df, batch_size,
                                                  loaders_dir + "train_loader{}.pth".format(i))
            break
        train_df = all_df.iloc[(len(all_df) // num_loaders) * i:(len(all_df) // num_loaders) * (i + 1), :]
        create_and_save_trainable_data_loader(tokenizer, train_df, batch_size,
                                              loaders_dir + "train_loader{}.pth".format(i))
