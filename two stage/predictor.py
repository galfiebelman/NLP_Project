import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loading import PREPOSITIONS, RELATIONS, ANSWERS


def evaluate(model_path, binary_model_path, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=len(
                                                                   RELATIONS))
    model.to(device)
    model.eval()
    binary_model = AutoModelForSequenceClassification.from_pretrained(
        binary_model_path, num_labels=len(ANSWERS))
    binary_model.to(device)
    binary_model.eval()
    test_df = pd.read_json("dev.jsonl", lines=True, orient='records')
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
                question = "Is there a relation between {} and {}?".format(
                    anchor, comp)
                sequence = tokenizer.encode_plus(question.lower(), text, return_tensors="pt")[
                    'input_ids'].to(device)
                with torch.no_grad():
                    out = binary_model(sequence)[0]
                probabilities = torch.softmax(out, dim=1).detach().cpu().tolist()[0]
                pred = np.argmax(np.array(probabilities))
                if int(pred) == 0:
                    cur_dict["predicted_prepositions"].append(int(pred))
                else:
                    question = "what is the relation between {} and {}?".format(anchor, comp)
                    sequence = tokenizer.encode_plus(question.lower(), text, return_tensors="pt")['input_ids'].to(device)
                    with torch.no_grad():
                        out = model(sequence)[0]
                    probabilities = torch.softmax(out, dim=1).detach().cpu().tolist()[0]
                    pred = np.argmax(np.array(probabilities))
                    cur_dict["predicted_prepositions"].append(int(pred)+1)
        dicts.append(cur_dict)

    with open(output_file, 'w') as out_file:
        for d in dicts:
            out_file.write(json.dumps(d))
            out_file.write("\n")
