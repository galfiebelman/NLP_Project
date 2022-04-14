import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import seaborn as sns
import matplotlib.pyplot as plt

from NLP_Project.one_stage.data_loading import create_and_save_trainable_data_loader, PREPOSITIONS
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


if __name__ == "__main__":
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)
    model_type = 'distilbert'
    loaders_dir = model_type + '_loaders/'
    load = False
    model_path = "model/" + model_type + '/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not load:
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                   num_labels=len(PREPOSITIONS))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(PREPOSITIONS))
    model.to(device)
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    epochs = 10
    grad_acc_steps = 1
    train_loader_paths = [os.path.join(loaders_dir, file) for file in
                          os.listdir(
                              "/home/gal/Private/University/Y1S1/NLP/Project/NLP_Project/one_stage/" + loaders_dir) if
                          "train_loader" in file]
    dev_loader_path = loaders_dir + "dev_loader.pth"
    train_losses, dev_accs = train_model(model, epochs, grad_acc_steps, optimizer, train_loader_paths, dev_loader_path,
                                         device, model_path)
    plot_train_dev_results(train_losses, dev_accs)
    print(train_losses)
    print(dev_accs)
