from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn

def train_model(
    model, train_loader, val_loader, device, epochs, lr,
    early_stopping_patience=10,
    target_accuracy=0.95,
    checkpoint_dir=None,
    resume_from=None,
    log_dir=None
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0

    # TensorBoard writer
    writer = None
    if log_dir is not None:
        writer = SummaryWriter(log_dir)

    # Resume checkpoint
    if resume_from is not None and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"Checkpoint caricato, riprendendo da epoch {start_epoch}")

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # Log su TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc
                }, os.path.join(checkpoint_dir, 'model_best.pth'))

        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping: non migliora da troppo tempo.")
                break

        if val_acc >= target_accuracy:
            print(f"Target accuracy {target_accuracy} raggiunta, stop training.")
            break

        if checkpoint_dir is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, os.path.join(checkpoint_dir, 'model_last.pth'))

    if writer is not None:
        writer.close()

    return model, train_losses, val_losses, train_accs, val_accs
