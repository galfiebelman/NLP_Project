from tqdm import tqdm
import torch
import numpy as np


def train_model(model, epochs, grad_acc_steps, optimizer, train_loader, dev_loader, device, save_path):
    train_loss_values = []
    dev_acc_values = []
    best_acc = 0
    for _ in tqdm(range(epochs), desc="Epoch"):

        # Training
        epoch_train_loss = 0  # Cumulative loss
        model.train()
        model.zero_grad()

        for step, batch in enumerate(train_loader):

            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)

            loss = outputs[0]
            loss = loss / grad_acc_steps
            epoch_train_loss += loss.item()

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:  # Gradient accumulation is over
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clipping gradients
                optimizer.step()
                model.zero_grad()

        epoch_train_loss = epoch_train_loss / len(train_loader)
        print("train loss:{}".format(epoch_train_loss))
        train_loss_values.append(epoch_train_loss)

        # Evaluation
        epoch_dev_accuracy = 0  # Cumulative accuracy
        model.eval()

        for batch in dev_loader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2]

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()

            predictions = np.argmax(logits, axis=1).flatten()
            labels = labels.numpy().flatten()

            epoch_dev_accuracy += np.sum(predictions == labels) / len(labels)

        epoch_dev_accuracy = epoch_dev_accuracy / len(dev_loader)
        if epoch_dev_accuracy > best_acc:
            best_acc = epoch_dev_accuracy
            print("saved")
            model.save_pretrained(save_path)
        print("dev acc:{}".format(epoch_dev_accuracy))
        dev_acc_values.append(epoch_dev_accuracy)

    return model, train_loss_values, dev_acc_values
