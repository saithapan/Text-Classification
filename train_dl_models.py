import torch

def train_dl_model(train_loader, val_loader, device,num_epochs, model, criterion, optimizer):


    for epoch in range(num_epochs):
        for i, (data_t, labels) in enumerate(train_loader):
            # Forward pass
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            data_t = data_t.to(device)
            outputs = model(data_t)
            # Compute loss
            loss = criterion(outputs, labels.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training progress
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (test_x, test_y) in val_loader:
                # break
                inputs = test_x.to(device)
                test_y = test_y.type(torch.LongTensor)
                labels = test_y.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%')

    return model