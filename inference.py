import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def predict(model, data_loader, device):

    predictions = []
    actual_values = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            preds = model(images).squeeze().cpu().numpy()
            predictions.extend(preds)
            actual_values.extend(labels.numpy())
    
    return predictions, actual_values

def compute_metrics(predictions, targets):
    predictions = torch.tensor(predictions, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    mse = nn.MSELoss()(predictions, targets).item()
    mae = nn.L1Loss()(predictions, targets).item()
    
    # Compute R-squared: 1 - SSE / SST
    sse = torch.sum((targets - predictions) ** 2).item()
    sst = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r_squared = 1 - sse / sst if sst > 0 else 0

    return mse, mae, r_squared

def plot_predictions(model, loader, device, denorm_fn=None, num_images=5):
    model.to(device)
    model.eval()

    images_shown = 0
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    with torch.no_grad():
        loader_iter = iter(loader)
        selected_batches = random.sample(range(len(loader)), min(len(loader), num_images // loader.batch_size + 1))

        for batch_idx in selected_batches:
            images, labels = next(loader_iter)

            batch_size = len(images)
            selected_indices = random.sample(range(batch_size), min(num_images - images_shown, batch_size))

            for idx in selected_indices:
                if images_shown >= num_images:
                    break

                image = images[idx].cpu()
                actual = labels[idx].item()

                if denorm_fn is not None:
                    image = denorm_fn(image)

                prediction = model(torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)).squeeze().cpu().item()

                ax = axes[images_shown]
                ax.imshow(image)
                ax.axis("off")
                ax.set_title(f"Actual: {actual:.2f}\nPredicted: {prediction:.2f}")

                images_shown += 1

            if images_shown >= num_images:
                break

    plt.show()

