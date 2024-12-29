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


def rotate_90(image_np, bboxes):
    """
    Rotates the image 90 degrees clockwise (top-left corner as the origin),
    and updates bounding boxes accordingly.

    image_np: (H, W, C) NumPy array
    bboxes: list of dicts => {
       'boundingBox': {'topLeftX','topLeftY','width','height'},
       'label': '(R,G,B)'
    }

    Returns:
      rotated_img_np: (W, H, C)  # dimensions swapped
      rotated_bboxes: same structure, but x,y,w,h for the rotated version
    """
    old_h, old_w = image_np.shape[:2]

    # Rotate image 90 deg clockwise => swap axes & flip
    # np.rot90(img, k=1) rotates 90 deg *counter-clockwise* by default.
    # So let's do this manually or do k=3 for clockwise.
    # For clarity, let's do manual approach:

    rotated_img_np = np.transpose(image_np, (1,0,2))  # (W,H,C)
    rotated_img_np = np.flipud(rotated_img_np)        # flip along vertical axis => 90 deg clockwise

    # Now we must rotate each bounding box 90 deg clockwise around top-left (0,0).
    # Original bounding box: (x, y, w, h)
    # After 90 deg clockwise:
    #   new_x = y
    #   new_y = old_w - x - w
    #   new_w = h
    #   new_h = w

    rotated_bboxes = []
    for rect in bboxes:
        x = rect['boundingBox']['topLeftX']
        y = rect['boundingBox']['topLeftY']
        w = rect['boundingBox']['width']
        h = rect['boundingBox']['height']

        new_x = y
        new_y = old_w - x - w
        new_w = h
        new_h = w

        rotated_bboxes.append({
            'boundingBox': {
                'topLeftX': new_x,
                'topLeftY': new_y,
                'width': new_w,
                'height': new_h
            },
            'label': rect['label']
        })

    return rotated_img_np, rotated_bboxes


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
        loss = 0.0
        for item in bbox_target_list:
            bbox = item['boundingBox']
            x1, y1 = bbox['topLeftX'], bbox['topLeftY']
            width, height = bbox['width'], bbox['height']

            # Parse label => (R,G,B)
            target_color = tuple(map(int, item['label'][1:-1].split(',')))
            target_class = self.get_the_class(target_color)

            # Crop from pred[0]
            pred_region = TF.crop(pred[0], top=y1, left=x1, height=height, width=width)
            pred_region = pred_region.squeeze(0)  # (H, W)

            target_tensor = torch.full(pred_region.shape,
                                       target_class,
                                       dtype=pred_region.dtype,
                                       device=pred_region.device)
            loss += self.mse_loss(pred_region, target_tensor)

        if len(bbox_target_list) > 0:
            return loss / len(bbox_target_list)
        else:
            return torch.tensor(0.0, device=pred.device)


class PlanDataset(Dataset):
    """
    This dataset returns *two items* for the single input image:
      0 => The original image + bounding boxes
      1 => The same image rotated 90 deg + bounding boxes
    """
    def __init__(self, image, rectangles):
        """
        image: PIL Image
        rectangles: list of dicts => {
          'boundingBox': {...}, 'label': '(R,G,B)'
        }
        """
        self.image = image
        self.rectangles = rectangles

        # Convert the PIL to numpy for convenience
        self.image_np = np.array(self.image)  # shape (H,W,C)

    def __len__(self):
        return 2  # We'll yield two samples: original + rotated

    def __getitem__(self, index):
        if index == 0:
            # 1) Original
            # Print bounding boxes
            print("\n[DATASET] Original bounding boxes:")
            for i, r in enumerate(self.rectangles):
                bb = r['boundingBox']
                print(f"  Box {i+1}: x={bb['topLeftX']}, y={bb['topLeftY']}, w={bb['width']}, h={bb['height']}  Label={r['label']}")
            # Convert to tensor
            plan_tensor = torch.from_numpy(self.image_np.transpose(2,0,1)).float()
            return plan_tensor, self.rectangles

        else:
            # 2) Rotated 90 deg
            rotated_img_np, rotated_bboxes = rotate_90(self.image_np, self.rectangles)

            print("\n[DATASET] Rotated bounding boxes:")
            for i, r in enumerate(rotated_bboxes):
                bb = r['boundingBox']
                print(f"  Box {i+1}: x={bb['topLeftX']}, y={bb['topLeftY']}, w={bb['width']}, h={bb['height']}  Label={r['label']}")

            plan_tensor = torch.from_numpy(rotated_img_np.transpose(2,0,1)).float()
            return plan_tensor, rotated_bboxes


def train_start(model, train_dataloader, bbox_target_list, device):
    """
    Example training loop. You might not use it if you're just printing.
    """
    criterion = CustomBBoxLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, betas=(0.9,0.999))

    model.train()
    for epoch in range(1):
        for idx, (img_batch, final_bboxes) in enumerate(train_dataloader):
            img_batch = img_batch.to(device)
            optimizer.zero_grad()
            pred = model(img_batch)
            loss = criterion(pred, final_bboxes)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Batch {idx+1}, Loss: {loss.item()}")
