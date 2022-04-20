import json
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelWithLMHead

from NLP_Project.one_stage.data_loading import PREPOSITIONS


def evaluate(model_path, output_file, tokenizer, acc):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(PREPOSITIONS))
    model.to(acc.device)
    model.eval()
    test_df = pd.read_json("test_unlabaled.jsonl", lines=True, orient='records')
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
