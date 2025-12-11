def predict_imu(model, loader, device=device):
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                X = batch['X'].to(device)
            elif isinstance(batch, (list, tuple)):
                X = batch[0].to(device)
            else:
                X = batch.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)  
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    return all_preds
