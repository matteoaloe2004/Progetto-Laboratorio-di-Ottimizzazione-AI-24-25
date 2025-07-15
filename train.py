import os
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, device, epochs,
                lr=0.001,
                early_stopping_patience=10,
                target_accuracy=None,
                checkpoint_dir=None,
                resume_from=None):
    """
    Training loop con:
    - resume da checkpoint
    - salvataggio best e last model
    - early stopping
    - stop su target accuracy
    
    Ritorna: modello allenato (e placeholder per compatibilità)
    """

    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    # Carica checkpoint se fornito
    if resume_from is not None and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Ripresa da checkpoint all'epoca {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # Salvataggio ultimo checkpoint
        if checkpoint_dir:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter
            }, os.path.join(checkpoint_dir, 'model_last.pth'))

        # Salvataggio miglior modello
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if checkpoint_dir:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_best.pth'))
            print(f"Nuovo miglior modello con val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping attivato all'epoca {epoch+1}")
            break

        # Stop su target accuracy
        if target_accuracy and val_acc >= target_accuracy:
            print(f"Target accuracy {target_accuracy} raggiunto, stop training")
            break

    return model, None, None, None, None
