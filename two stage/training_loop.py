from tqdm import tqdm
import torch
import numpy as np


def train_model(model, epochs, grad_acc_steps, optimizer, train_loader, dev_loader, device):
    train_loss_values = []
    dev_acc_values = []

    for _ in tqdm(range(epochs), desc="Epoch"):

        # Training
        epoch_train_loss = 0  # Cumulative loss
        model.train()
        model.zero_grad()

        for step, batch in enumerate(train_loader):

            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)

            loss = outputs[0]
            loss = loss / grad_acc_steps
            epoch_train_loss += loss.item()

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:  # Gradient accumulation is over
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clipping gradients
                optimizer.step()
                model.zero_grad()

        epoch_train_loss = epoch_train_loss / len(train_loader)
        train_loss_values.append(epoch_train_loss)

        # Evaluation
        epoch_dev_accuracy = 0  # Cumulative accuracy
        model.eval()

        for batch in dev_loader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2]

            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()

            predictions = np.argmax(logits, axis=1).flatten()
            labels = labels.numpy().flatten()

            epoch_dev_accuracy += np.sum(predictions == labels) / len(labels)

        epoch_dev_accuracy = epoch_dev_accuracy / len(dev_loader)
        dev_acc_values.append(epoch_dev_accuracy)

    return model, train_loss_values, dev_acc_values
