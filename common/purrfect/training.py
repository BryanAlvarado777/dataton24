import torch
from tqdm import tqdm  # Progress bar
from purrfect.metrics import MetricAccumulator, WMAPE, DICEE, DPEAKS
import os


# Combined train/validate function with progress bar
def train_validate(
    model,
    loader,
    criterion,
    optimizer=None,
    epoch=None,
    metric_accumulator=None,
    device="cuda",
):
    """
    Train or validate the model depending on whether an optimizer is provided, with a progress bar.

    Args:
        model (nn.Module): The model to be trained or validated.
        loader (DataLoader): DataLoader for the training/validation data.
        criterion (nn.Module): The loss function (Focal Loss).
        optimizer (torch.optim.Optimizer or None): Optimizer for training. If None, the function only validates.
        epoch (int or None): Current epoch number for display (optional).
        metric_accumulator (dict or None): Dictionary with metrics functions to be used for validation

    Returns:
        avg_loss (float): The average loss over the dataset.
    """
    if optimizer:
        model.train()  # Set the model to training mode
        mode = "Train"
    else:
        model.eval()  # Set the model to evaluation mode
        mode = "Validate"
    if metric_accumulator:
        metric_accumulator.reset()

    total_loss = 0.0

    # Create a progress bar for the loop
    loop = tqdm(
        enumerate(loader), desc=f"{mode} Epoch {epoch}", leave=True, total=len(loader)
    )

    # Disable gradient calculation for validation
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            # print(f"Batch {i+1}/{len(loader)}")
            # Move data to the device (GPU if available)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            # print("antes forward")
            outputs = model(inputs)
            # print("despues forward")
            loss = criterion(outputs, targets)
            if metric_accumulator:
                metric_accumulator.update(
                    targets.squeeze(1).detach().cpu(), outputs.squeeze(1).detach().cpu()
                )
            # Backward pass and optimization (if training)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()
            # Update progress bar description with current loss
            loop.update(1)
            loop.set_postfix(loss=loss.item())

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(loader)
    metrics = dict()
    if metric_accumulator:
        metrics = metric_accumulator.get_metrics()
    metrics["Loss"] = avg_loss
    loop.set_postfix(metrics)
    loop.refresh()
    return avg_loss


# Main function for the training process
def train_model(
    model,
    train_loader,
    val_loader,
    best_model_path,
    last_checkpoint_path,
    criterion,
    optimizer,
    num_epochs=10,
    device="cuda",
    early_stopping_patience=None,
):
    """
    Train and validate the model using Adam optimizer and Focal Loss.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the Adam optimizer.
        gamma (float): Focal loss gamma parameter.
        alpha (float): Focal loss alpha parameter.

    Returns:
        None
    """
    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, weights_only=False)
        best_loss = checkpoint["best_loss"]
        init_epoch = checkpoint["epoch"] + 1
        epochs_since_last_improvement = checkpoint["epochs_since_last_improvement"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        init_epoch = 1
        best_loss = float("inf")
        epochs_since_last_improvement = 0

    metric_accumulator = MetricAccumulator.create_default()
    # Loop through epochs
    for epoch in range(init_epoch, num_epochs + 1):
        if early_stopping_patience:
            if epochs_since_last_improvement == early_stopping_patience:
                print(
                    f"early stopping: {epochs_since_last_improvement} epochs without improvement"
                )
                break
        print(f"Epoch [{epoch}/{num_epochs}]")

        # Train the model
        train_validate(
            model, train_loader, criterion, optimizer, epoch, metric_accumulator, device
        )

        # Validate the model
        validate_loss = train_validate(
            model,
            val_loader,
            criterion,
            epoch=epoch,
            metric_accumulator=metric_accumulator,
            device=device,
        )
        if validate_loss <= best_loss:
            torch.save(model.state_dict(), best_model_path)
            print("Saving best model")
            best_loss = validate_loss
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "epochs_since_last_improvement": epochs_since_last_improvement,
                "epoch": epoch,
            },
            last_checkpoint_path,
        )

    print("Training complete.")
