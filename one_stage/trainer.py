import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import seaborn as sns
import matplotlib.pyplot as plt

from NLP_Project.one_stage.data_loading import create_trainable_data_loader, PREPOSITIONS
from NLP_Project.one_stage.training_loop import train_model


def plot_train_dev_results(train_losses, dev_accs):
    sns.set()

    plt.plot(train_losses, label="train_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.xticks(np.arange(len(train_losses)))
    plt.savefig("train_loss.png")
    # plt.show()
    #
    # plt.waitforbuttonpress(timeout=1)

    plt.plot(dev_accs, label="dev_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy")
    plt.legend()
    plt.xticks(np.arange(len(dev_accs)))
    plt.savefig("dev_acc.png")
    # plt.show()


def train(save_path, train_loader_save_path, dev_loader_save_path, train_loader_load_path, dev_loader_load_path):
    max_seq_length = 256
    batch_size = 40
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    assert ((dev_loader_load_path is None and train_loader_load_path is None) or (
                dev_loader_load_path is not None and train_loader_load_path is not None))
    assert ((dev_loader_save_path is None and train_loader_save_path is None) or (
            dev_loader_save_path is not None and train_loader_save_path is not None))

    if train_loader_load_path is None:
        all_df = pd.read_json('train.jsonl', lines=True, orient='records')
        train_df = all_df.head(1000)
        dev_df = all_df.tail(100)
        print("creating loaders")
        train_loader = create_trainable_data_loader(tokenizer, train_df, max_seq_length, batch_size)
        print("trainloader done")
        dev_loader = create_trainable_data_loader(tokenizer, dev_df, max_seq_length, batch_size)
        print("devloader done")

    else:
        train_loader = torch.load(train_loader_load_path)
        dev_loader = torch.load(dev_loader_load_path)

    if train_loader_save_path is not None and train_loader_load_path is None:
        torch.save(train_loader, train_loader_save_path)
        torch.save(dev_loader, dev_loader_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(PREPOSITIONS))

    model.to(device)
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    epochs = 3
    grad_acc_steps = 1

    model, train_losses, dev_accs = train_model(model, epochs, grad_acc_steps, optimizer, train_loader, dev_loader,
                                                device, save_path)
    plot_train_dev_results(train_losses, dev_accs)
    print(train_losses)
    print(dev_accs)
