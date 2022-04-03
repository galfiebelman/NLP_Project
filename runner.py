import random
import torch
import numpy as np
import pandas as pd

from NLP_Project.predictor import evaluate
from NLP_Project.scoring import score
from NLP_Project.trainer import train


if __name__ == "__main__":
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)
    train_loader_load_path = 'train_loader.pth'
    dev_loader_load_path = 'dev_loader.pth'
    train_loader_save_path = 'train_loader.pth'
    dev_loader_save_path = 'dev_loader.pth'
    output_file = 'eval_predictions.jsonl'
    eval_file = 'dev_eval.jsonl'
    score_file = 'score.jsonl'
    model_path = "model/"
    train(model_path, train_loader_save_path, dev_loader_save_path, train_loader_load_path, dev_loader_load_path)
    evaluate(model_path, output_file)
    score(output_file, eval_file, score_file)

