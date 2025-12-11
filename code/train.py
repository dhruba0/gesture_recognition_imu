def train_model(model, train_loader,criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for p in train_loader:
            
            inputs = p['X'].to(device)            
            targets = p['y'].to(device)      

            optimizer.zero_grad()
            outputs = model(inputs)              
            loss = loss = criterion(F.log_softmax(outputs, dim=1), targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            # print(targets)
            # print(predicted)
            _,targets = torch.max(targets,1)
            correct += (predicted == targets).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
