import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn import metrics
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm
from pytorchtools import EarlyStopping

# patience = 50
# early_stopping = EarlyStopping(patience, verbose=True)

def do_compute_metrics(probas_pred, target):
    # pred = (probas_pred >= 0.5).astype(int)
    probas_pred = np.array(probas_pred)
    pred = [1 if i else 0 for i in (probas_pred >= 0.5)]
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


class Trainer(object):
    def __init__(self, model, optim, device, label_coefficient, label_text, train_dataloader, val_dataloader, test_dataloader,
                 discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.k = 0.05
        self.extent = 0.1
        self.label_coefficient = torch.tensor(label_coefficient, dtype=torch.float).to(self.device)
        self.label_text = label_text.to(self.device)
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = model
        self.best_epoch = None
        self.best_auroc = 0
        self.best_acc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "F1", "Accuracy", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Accuracy", "Test_loss"]

        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)


    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            alpha = self.extent * np.exp(-self.k * i)
            self.current_epoch += 1
            train_loss = self.train_epoch(alpha)
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            if self.experiment:
                self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, f1, accuracy, val_loss = self.test(alpha, dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, f1, accuracy, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if accuracy >= self.best_acc:
                self.best_model = copy.deepcopy(self.model)
                self.best_acc = accuracy
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc) + " Accuracy " + str(accuracy) + " F1 " + str(f1))

            # early_stopping(val_loss, self.model)
            #
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #
            #     break

        auroc, auprc, f1, accuracy, test_loss = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, accuracy, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Accuracy " + str(accuracy) + " F1 " + str(f1))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_max_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }

        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())


    def train_epoch(self, alpha):
        self.model.train()
        label_text = self.label_text
        loss_epoch_1 = 0
        loss_epoch_2 = 0

        num_batches = len(self.train_dataloader)
        for i, (graph, y, label, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            y, label, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description = y.float().to(self.device), label.to(self.device), v_d_target.to(
                self.device), v_d_smile.to(self.device), v_d_description.to(self.device), v_p_target.to(
                self.device), v_p_smile.to(
                self.device), v_p_description.to(self.device)
            graph = [tensor.to(device=self.device) for tensor in graph]
            self.optim.zero_grad()
            score, loss_2 = self.model(graph, label, self.label_coefficient, alpha, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description, label_text)
            if self.n_class == 1:
                n, loss_1 = binary_cross_entropy(score, y)
            else:
                n, loss_1 = binary_cross_entropy(score, y)
            loss = loss_1 + loss_2
            loss.backward()
            self.optim.step()
            loss_epoch_1 += loss_1.item()
            loss_epoch_2 += loss_2
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)

        loss_epoch_1 = loss_epoch_1 / num_batches
        loss_epoch_2 = loss_epoch_2 / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss_1 ' + str(loss_epoch_1) + ' with training loss_2 ' + str(loss_epoch_2))
        return loss_epoch_1 + loss_epoch_2


    def test(self, alpha=0.0, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            label_text = self.label_text
            for i, (graph, y, label, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description) in enumerate(tqdm(data_loader)):
                y, label, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description = y.float().to(self.device), label.to(self.device), v_d_target.to(
                    self.device), v_d_smile.to(self.device), v_d_description.to(self.device), v_p_target.to(
                    self.device), v_p_smile.to(self.device), v_p_description.to(self.device)
                graph = [tensor.to(device=self.device) for tensor in graph]
                if dataloader == "val":
                    score, loss_2 = self.model(graph, label, self.label_coefficient, alpha, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description, label_text)
                elif dataloader == "test":
                    score, loss_2 = self.best_model(graph, label, self.label_coefficient, alpha, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description, label_text)
                if self.n_class == 1:
                    n, loss_1 = binary_cross_entropy(score, y)
                else:
                    n, loss = cross_entropy_logits(score, y)
                test_loss += loss_1.item()
                test_loss += loss_2
                y_label = y_label + y.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

        test_loss = test_loss / num_batches
        accuracy, auroc, f1, precision, recall, int_ap, auprc = do_compute_metrics(y_pred, y_label)

        return auroc, auprc, f1, accuracy, test_loss

