import albumentations as A
import torchvision.transforms.functional as TF
from albumentations.pytorch.transforms import ToTensorV2  # optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

albumentations_transform = A.Compose(
    [
        A.Rotate(limit=(90, 90), p=1.0), 
    ],
    bbox_params=A.BboxParams(
        format='coco',         # we have [x, y, width, height]
        label_fields=['labels'],
        min_area=1,
        min_visibility=0.1
    )
)

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

        for class_id, mapped_color in color_mapping.items():
            if color == mapped_color:
                return class_id

        raise ValueError(f"Color {color} does not match any known class.")

    def forward(self, pred, bbox_target_list):
        # pred => (B,1,H,W). We'll assume B=1 for your single image
        loss = 0.0

        for item in bbox_target_list:
            bbox = item['boundingBox']
            x1, y1 = bbox['topLeftX'], bbox['topLeftY']
            width, height = bbox['width'], bbox['height']

            # Parse (R,G,B)
            target_color = tuple(map(int, item['label'][1:-1].split(',')))
            target_class = self.get_the_class(target_color)

            # Crop out region from pred[0]
            pred_region = TF.crop(pred[0], top=y1, left=x1, height=height, width=width)
            pred_region = pred_region.squeeze(0)  # shape => (regionH, regionW)

            # Build the target tensor
            target_tensor = torch.full(
                pred_region.shape,
                target_class,
                dtype=pred_region.dtype,
                device=pred_region.device
            )

            loss += self.mse_loss(pred_region, target_tensor)

        if len(bbox_target_list) > 0:
            return loss / len(bbox_target_list)
        else:
            return torch.tensor(0.0, device=pred.device)


class PlanDataset(Dataset):
    def __init__(self, image, rectangles, transform=None):
        self.image = image
        self.rectangles = rectangles
        self.transform = transform

    def __len__(self):
        # For demo, we'll assume 1 item
        return 1

    def __getitem__(self, index):
        # Convert the PIL image to a NumPy array (H, W, C)
        image_np = np.array(self.image)

        # Prepare bounding boxes in COCO format: [x_min, y_min, width, height]
        bboxes = []
        labels = []
        for rect in self.rectangles:
            box = rect['boundingBox']
            x, y = box['topLeftX'], box['topLeftY']
            w, h = box['width'], box['height']
            bboxes.append([x, y, w, h])
            labels.append(rect.get('label', 'No label'))

        if self.transform is not None:
            transformed = self.transform(
                image=image_np,
                bboxes=bboxes,
                labels=labels
            )
        else:
            # No transform => just pass them through
            transformed = {
                'image': image_np,
                'bboxes': bboxes,
                'labels': labels
            }

        transformed_image = transformed['image']    # NumPy array (H, W, C)
        transformed_bboxes = transformed['bboxes']  # list of [x, y, w, h]
        transformed_labels = transformed['labels']  # list of labels

        # Convert the (H, W, C) array into a Torch tensor (C, H, W)
        plan_tensor = torch.from_numpy(transformed_image.transpose(2, 0, 1)).float()

        # Reformat bounding boxes and labels into the requested structure
        formatted_bboxes = []
        for box, label in zip(transformed_bboxes, transformed_labels):
            x, y, w, h = box
            formatted_bboxes.append({
                'boundingBox': {
                    'topLeftX': x,
                    'topLeftY': y,
                    'width': w,
                    'height': h
                },
                'label': label
            })

        # Return the image tensor and the new list of dicts
        return plan_tensor, formatted_bboxes


def train_start(model, train_dataloader, bbox_target_list, device):
    criterion = CustomBBoxLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, betas=(0.9,0.999))

    model.train()
    for epoch in range(1):  # Single epoch
        for idx, (img_batch, final_bboxes) in enumerate(train_dataloader):
            img_batch = img_batch.to(device)

            optimizer.zero_grad()
            pred = model(img_batch)  # shape => (B,1,H,W)

            loss = criterion(pred, final_bboxes)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {idx+1}, Loss: {loss.item()}")
