import torch.nn as nn
import torchvision.models as models

def create_cnn_model(num_classes=38):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),

        nn.Flatten(),
        nn.Linear(64 * 56 * 56, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

def create_transfer_model(num_classes=38, fine_tune_at=100):
    base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in base_model.features[:fine_tune_at].parameters():
        param.requires_grad = False

    base_model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base_model.last_channel, num_classes)
    )

    return base_model
