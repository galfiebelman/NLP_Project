from tqdm import tqdm
import torch
import numpy as np


def train_model(model, epochs, grad_acc_steps, optimizer, train_loader, dev_loader, acc, model_path):
    train_loss_values = []
    dev_acc_values = []
    best_acc = 0
    cnt = 0
    for _ in tqdm(range(epochs), desc="Epoch"):

        epoch_train_loss = 0
        model.train()
        model.zero_grad()

        for step, batch in enumerate(train_loader):

            input_ids = batch[0]
            attention_masks = batch[1]
            labels = batch[2]

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)

            loss = outputs[0]
            loss = loss / grad_acc_steps
            epoch_train_loss += loss.item()

            acc.backward(loss)

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()

        epoch_train_loss = epoch_train_loss / len(train_loader)
        print("train loss:{}".format(epoch_train_loss))
        train_loss_values.append(epoch_train_loss)
        epoch_dev_accuracy = 0
        model.eval()

        for batch in dev_loader:
            input_ids = batch[0]
            attention_masks = batch[1]
            labels = batch[2]

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()

            predictions = np.argmax(logits, axis=1).flatten()
            labels = labels.detach().cpu().numpy().flatten()

            epoch_dev_accuracy += np.sum(predictions == labels) / len(labels)

        epoch_dev_accuracy = epoch_dev_accuracy / len(dev_loader)
        print("dev acc:{}".format(epoch_dev_accuracy))
        dev_acc_values.append(epoch_dev_accuracy)
        if epoch_dev_accuracy > best_acc:
            best_acc = epoch_dev_accuracy
            print("saved")
            model.save_pretrained(model_path)
            cnt = 0
        else:
            cnt += 1
            if cnt == epochs // 4:
                return train_loss_values, dev_acc_values

    return train_loss_values, dev_acc_values
