import torch
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay


def train_model(model, criterion, optimizer, dataloaders, image_datasets, patience=0, num_epochs=3):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    last_loss = 200
    triggertimes = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))

            # Early stopping
            if phase == 'train' or patience <= 0:
                continue

            if epoch_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    return best_model
            else:
                trigger_times = 0
                best_model = model

            last_loss = epoch_loss

    return best_model


def test(model, test_dataset):
    x_test = [i for i, j in test_dataset]
    y_test = [j for i, j in test_dataset]
    preds = []
    outs = []
    for t in x_test:
        pred, out = predict(model, t)
        preds.append(pred)

    return accuracy_score(preds, y_test), f1_score(preds, y_test)


def predict(model, x, ):
    model.eval()
    out = model(x.reshape(1, 1, 128, 128))
    _, prediction = torch.max(out, dim=1)
    return prediction[0].item(), out