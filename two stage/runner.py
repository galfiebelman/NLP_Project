import random
import torch
import numpy as np
import pandas as pd

from predictor import evaluate
from scoring import score
from trainer import train


if __name__ == "__main__":
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)
    train_loader_load_path = 'train_loader.pth'
    dev_loader_load_path = 'dev_loader.pth'
    train_loader_save_path = 'train_loader.pth'
    dev_loader_save_path = 'dev_loader.pth'
    binary_train_loader_load_path = 'binary_train_loader.pth'
    binary_dev_loader_load_path = 'binary_dev_loader.pth'
    binary_train_loader_save_path = 'binary_train_loader.pth'
    binary_dev_loader_save_path = 'binary_dev_loader.pth'
    output_file = 'eval_predictions.jsonl'
    eval_file = 'dev_eval.jsonl'
    score_file = 'score.jsonl'
    model_path = "model/"
    binary_model_path = "binary_model/"
    train(model_path, train_loader_save_path, dev_loader_save_path,
          train_loader_load_path, dev_loader_load_path, False)
    train(model_path, train_loader_save_path, dev_loader_save_path,
          train_loader_load_path, dev_loader_load_path, True)
    evaluate(model_path, output_file)
    score(output_file, eval_file, score_file)

