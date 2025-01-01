import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from PIL import Image
import numpy as np

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

        # Raise an error if no match
        raise ValueError(f"Color {color} does not match any known class.")

    def forward(self, pred, bbox_target_list):
        loss = 0.0

        for item in bbox_target_list:
            # Extract bounding box coordinates
            bbox = item['boundingBox']
            x1 = int(round(bbox['topLeftX']))
            y1 = int(round(bbox['topLeftY']))
            width = int(round(bbox['width']))
            height = int(round(bbox['height']))

            # Parse and map the target color to its class
            target_color = tuple(map(int, item['label'][1:-1].split(',')))  # Convert '(R,G,B)' to (R, G, B)
            target_class = self.get_the_class(target_color)

            # Extract the corresponding region from the prediction
            pred_region = TF.crop(pred, top=y1, left=x1, height=height, width=width) 

            pred_region = pred_region.squeeze(0) 

            # Create the target tensor
            target_tensor = torch.full(
                pred_region.shape,
                target_class,
                dtype=pred_region.dtype,
                device=pred_region.device
            )

            loss += self.mse_loss(pred_region, target_tensor) # MSE loss for the current bbox

        return loss / len(bbox_target_list)  # Normalize loss, then return

class OutsideLoss(nn.Module):
    def __init__(self):
        super(OutsideLoss, self).__init__()
        self.l1_loss = nn.MSELoss() 

    def forward(self, pred, bbox_target_list):
        # Create a binary mask for bounding box areas
        mask = torch.zeros_like(pred, dtype=torch.bool)  # Initialize mask with False (no areas selected)

        for item in bbox_target_list:
            # Extract bounding box coordinates
            bbox = item['boundingBox']
            x1, y1 = bbox['topLeftX'], bbox['topLeftY']
            width, height = bbox['width'], bbox['height']

            # Mark bounding box areas in the mask
            mask[:, y1:y1+height, x1:x1+width] = True

        # Create the complementary mask for areas outside the bounding boxes
        outside_mask = ~mask

        # Extract predictions and ground truth values for the outside regions
        outside_pred = pred[outside_mask]
        outside_target = pred[outside_mask].detach()  # Use current predictions as the "unchanged" target

        # Compute L1 loss for the outside regions
        loss_outside = self.l1_loss(outside_pred, outside_target)

        return loss_outside

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

def augment_img_bbox(original_image, extracted_data):

    alb_bboxes = []
    alb_labels = []
    for rect in extracted_data:
        bb = rect['boundingBox']
        label = rect['label']
        
        x = bb['topLeftX']
        y = bb['topLeftY']
        w = bb['width']
        h = bb['height']
        
        alb_bboxes.append([x, y, w, h])
        alb_labels.append(label)

    transform_90 = A.Compose(
        [
            A.Rotate(limit=[90, 90], p=1.0)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
    )
    augmented = transform_90(
        image=np.array(original_image),
        bboxes=alb_bboxes,
        labels=alb_labels
    )
    # Convert augmented image (NumPy array) back to PIL
    aug_image = Image.fromarray(augmented['image'])
    aug_bboxes = augmented['bboxes']  # still in COCO format => [x, y, w, h]

    aug_bboxes_data = []
    for bbox, label in zip(aug_bboxes, alb_labels):
        x, y, w, h = bbox
        aug_bboxes_data.append({
            'boundingBox': {
                'topLeftX': int(x),
                'topLeftY': int(y),
                'width': int(w),
                'height': int(h)
            },
            'label': label
        })
    return aug_image, aug_bboxes_data

def train_start(model, train_dataloader, bboxes_list, device):
    criterion = CustomBBoxLoss()
    criterion_outside = OutsideLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-6,  # lowered for stability
        betas=(0.9, 0.999),
    )

    model.train()

    for epoch in range(1):  # Single epoch
        for idx, img_batch in enumerate(train_dataloader):

            img_batch = img_batch.to(device) # move to the device

            bbox_targets = bboxes_list[idx] # bbox for the current image

            optimizer.zero_grad() # reset gradients

            pred = model(img_batch) # make a prediction

            loss = criterion(pred, bbox_targets) # loss for boxes
    
            loss_outside = criterion_outside(pred, bbox_targets) # loss for outside boxes

            total_loss = loss + 0.1 * loss_outside # total loss
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Batch {idx + 1}, Loss: {loss.item()}")
