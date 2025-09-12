"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from pathlib import Path
from torchvision import transforms
from timeit import default_timer as timer
from torchvision.transforms import TrivialAugmentWide

#Setup saving directory
SAVE_DIR=Path("Save_dir_100ep_dm768_lre5_ly12_hd12")
SAVE_DIR.mkdir(parents=True, exist_ok=True)  # Creates the directory if it does not exist

# Setup hyperparameters
NUM_EPOCHS =100
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
D_MODEL= 768
IMAGE_SIZE=(224,224)
PATCH_SIZE=(16,16)
N_CHANNELS=3
N_HEADS=12
N_LAYERS=12
LAYER_MLP=4*D_MODEL # Neuron in each layer of mlp

# Setup directories
train_dir = "/media/sujata/G/Datasets/SoyMCDataset/train"
test_dir = "/media/sujata/G/Datasets/SoyMCDataset/val"              #Note: this is the validation directory
real_test_dir = "/media/sujata/G/Datasets/SoyMCDataset/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  TrivialAugmentWide(num_magnitude_bins=31),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, real_test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    real_test_dir=real_test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model=model_builder.VisionTransformer(
    d_model=D_MODEL,
    n_classes=len(class_names),
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    n_channels=N_CHANNELS,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    layer_mlp=LAYER_MLP
)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the timer 
start_time=timer()

# Start training with help from engine.py
engine.train_and_test(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             real_test_dataloader=real_test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             class_names=class_names,
             save_dir=SAVE_DIR
             )

# End the timer
end_time=timer()
experimentation_time=end_time-start_time

print(f"\nTotal Experimentation time: {utils.format_time(experimentation_time)}")

#saving model summary using save_model_summary.py

utils.save_model_summary(model,input_size=[32,3,224,224], file_path=SAVE_DIR/"model_summary.txt", use_torchinfo=True, device=device)

print(f"\nModel_summary saved to {SAVE_DIR}/model_summary.txt ")
print(f"\n--------------------------------------------------------------------------------------------------------------------")

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir=SAVE_DIR,
                 model_name="300_epoch_ly1_lr1e4_dm768_hd12_ViT_base_model.pth")


print(f"\n----------------------------------------------------------------------------------------------------------------------")
print(f"\nExperiment completed !!! ")

