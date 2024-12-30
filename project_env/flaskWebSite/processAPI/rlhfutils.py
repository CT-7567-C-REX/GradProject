import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms.functional as TF

class CustomBBoxLoss(nn.Module):
    def __init__(self):
        super(CustomBBoxLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def get_the_class(self, color):
        color_mapping = {
            0: (0, 0, 0),       # Walls
            1: (255, 80, 80),   # Iwan
            2: (80, 80, 255),   # Room
            3: (255, 255, 0),   # Stairs
            4: (255, 255, 255), # Background
        }

        # Find the class
        for class_id, mapped_color in color_mapping.items():
            if color == mapped_color:
                return class_id

        # Raise an error if color is invalid
        raise ValueError(f"Color {color} does not match any known class.")

    def forward(self, pred, bbox_target_list):
        loss = 0.0

        for item in bbox_target_list:
            # Extract bounding box coordinates
            bbox = item['boundingBox']
            x1, y1 = bbox['topLeftX'], bbox['topLeftY']
            width, height = bbox['width'], bbox['height']

            # Parse and map the target color to its class
            target_color = tuple(map(int, item['label'][1:-1].split(',')))  # Convert '(R,G,B)' to (R, G, B)
            target_class = self.get_the_class(target_color)

            # Extract the corresponding region from the prediction
            pred_region = TF.crop(pred, top=y1, left=x1, height=height, width=width)  # shape: (1, H, W)
            #pred_region = torch.round(pred_region)

            # Since `pred` is single-channel, squeeze to simplify
            pred_region = pred_region.squeeze(0)  # shape: (H, W)

            # Create the target tensor with the same shape as pred_region
            target_tensor = torch.full(
                pred_region.shape,
                target_class,
                dtype=pred_region.dtype,
                device=pred_region.device
            )

            # Compute MSE loss for the current bounding box
            loss += self.mse_loss(pred_region, target_tensor) 

        # Normalize the loss by the number of bounding boxes
        return loss / len(bbox_target_list)
    
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class PlanDataset(Dataset):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        # 1) Get the image at the specified index
        pil_image = self.images_list[index]
        
        # 2) Convert to NumPy array (float32)
        plan = np.array(pil_image).astype(np.float32)

        # 3) If a transform is provided, apply it
        if self.transform is not None:
            transformed = self.transform(image=plan)
            plan = transformed['image']

        # 4) Convert from (H, W, C) to (C, H, W) and make a tensor
        plan_tensor = torch.from_numpy(plan.transpose((2, 0, 1)))

        # 5) Return the image tensor
        return plan_tensor



def train_start(model, train_dataloader, bboxes_list, device):
    criterion = CustomBBoxLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-6,  # lowered for stability
        betas=(0.9, 0.999),
    )

    model.train()

    for epoch in range(1):  # Single epoch
        for idx, img_batch in enumerate(train_dataloader):
            # Move image batch to GPU/CPU device
            img_batch = img_batch.to(device)

            # 1) Retrieve the bounding boxes for this index
            #    Since batch_size=1, idx matches image index
            bbox_targets = bboxes_list[idx]

            # 2) Zero out the gradients
            optimizer.zero_grad()

            # 3) Forward pass through the model
            pred = model(img_batch)

            # 4) Compute the loss using your custom criterion
            loss = criterion(pred, bbox_targets)

            # 5) Backpropagate and update
            loss.backward()
            optimizer.step()

            # 6) Print loss
            print(f"Epoch {epoch + 1}, Batch {idx + 1}, Loss: {loss.item()}")
