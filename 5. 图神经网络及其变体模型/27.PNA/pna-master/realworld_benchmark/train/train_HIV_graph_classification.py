import torch
from ogb.graphproppred import Evaluator

def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    list_scores = []
    list_labels = []
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        list_scores.append(batch_scores.detach())
        list_labels.append(batch_labels.detach().unsqueeze(-1))

    epoch_loss /= (iter + 1)
    evaluator = Evaluator(name='ogbg-molhiv')
    epoch_train_ROC = evaluator.eval({'y_pred': torch.cat(list_scores),
                                       'y_true': torch.cat(list_labels)})['rocauc']

    return epoch_loss, epoch_train_ROC, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_ROC = 0
    with torch.no_grad():
        list_scores = []
        list_labels = []
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_scores = model.forward(batch_graphs, batch_x)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            list_scores.append(batch_scores.detach())
            list_labels.append(batch_labels.detach().unsqueeze(-1))

        epoch_test_loss /= (iter + 1)
        evaluator = Evaluator(name='ogbg-molhiv')
        epoch_test_ROC = evaluator.eval({'y_pred': torch.cat(list_scores),
                                           'y_true': torch.cat(list_labels)})['rocauc']

    return epoch_test_loss, epoch_test_ROC
