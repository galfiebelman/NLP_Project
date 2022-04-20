import random

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

from NLP_Project.one_stage.data_loading import create_and_save_trainable_data_loader, PREPOSITIONS
from NLP_Project.one_stage.predictor import evaluate
from NLP_Project.one_stage.training_loop import train_model

if __name__ == "__main__":
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)
    accelerator = Accelerator()
    batch_size = 8
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_df = pd.read_json('train.jsonl', lines=True, orient='records')
    dev_df = pd.read_json('dev.jsonl', lines=True, orient='records')
    train_loader = create_and_save_trainable_data_loader(tokenizer, dev_df, batch_size)
    dev_loader = create_and_save_trainable_data_loader(tokenizer, dev_df, batch_size)
    model_path = 'model/'
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=len(PREPOSITIONS))
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    epochs = 40
    grad_acc_steps = 4
    model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)
    train_losses, dev_accs = train_model(model, epochs, grad_acc_steps, optimizer, train_loader, dev_loader,
                                         accelerator, model_path)
    print(train_losses)
    print(dev_accs)
    output_file = 'predictions.jsonl'
    evaluate(model_path, output_file, tokenizer, accelerator)
