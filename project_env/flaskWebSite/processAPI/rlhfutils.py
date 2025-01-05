import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from PIL import Image
import numpy as np
import random
class CustomBBoxLoss(nn.Module):
    def __init__(self):
        super(CustomBBoxLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def get_the_class(self, color):
        color_mapping = {
            0: (0, 0, 0),       # Walls
            1: (80, 80, 255),   # Room
            2: (255, 80, 80),   # Iwan
            3: (255, 255, 0),   # Stairs
            4: (255, 255, 255), # Background
        }

        for class_id, mapped_color in color_mapping.items():
            if color == mapped_color:
                return class_id

        raise ValueError(f"Color {color} does not match any known class.")

    def forward(self, pred, bbox_target_list):
        loss = 0.0
        gradient_mask = torch.zeros_like(pred, dtype=torch.float32, device=pred.device)

        for item in bbox_target_list:
            bbox = item['boundingBox']
            x1 = int(round(bbox['topLeftX']))
            y1 = int(round(bbox['topLeftY']))
            width = int(round(bbox['width']))
            height = int(round(bbox['height']))

            target_color = tuple(map(int, item['label'][1:-1].split(',')))
            target_class = self.get_the_class(target_color)

            pred_region = TF.crop(pred, top=y1, left=x1, height=height, width=width)
            pred_region = pred_region.squeeze(0)

            target_tensor = torch.full(
                pred_region.shape,
                target_class,
                dtype=pred_region.dtype,
                device=pred_region.device
            )

            loss += self.mse_loss(pred_region, target_tensor)

            # Update the gradient mask for this bounding box
            gradient_mask[:, :, y1:y1 + height, x1:x1 + width] = 1.0

        # Apply gradient masking
        pred.register_hook(lambda grad: grad * gradient_mask)

        return loss / len(bbox_target_list)

class OutsideLoss(nn.Module):
    def __init__(self):
        super(OutsideLoss, self).__init__()
        self.l1_loss = nn.MSELoss()

    def forward(self, pred, pred_image, bbox_target_list):
        pred_image = np.array(pred_image).astype(np.float32) / 255.0 * 4
        if pred_image.ndim == 3 and pred_image.shape[2] == 3:
            pred_image = pred_image[..., 0]
        pred_image_tensor = torch.from_numpy(pred_image).unsqueeze(0).unsqueeze(0).to(pred.device)

        mask = torch.zeros_like(pred, dtype=torch.bool)

        for item in bbox_target_list:
            bbox = item['boundingBox']
            x1, y1 = int(bbox['topLeftX']), int(bbox['topLeftY'])
            width, height = int(bbox['width']), int(bbox['height'])
            mask[:, :, y1:y1 + height, x1:x1 + width] = True

        outside_mask = ~mask
        outside_pred = pred[outside_mask]
        outside_target = pred_image_tensor[outside_mask]

        return self.l1_loss(outside_pred, outside_target)


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

def train_start(model, train_dataloader, pred_image, bboxes_list, device):
    criterion = CustomBBoxLoss()
    criterion_outside = OutsideLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-6,
        betas=(0.9, 0.999),
    )

    layers = [model.inc, model.down1, model.down2, model.up3, model.up4, model.out]

    model.train()

    for epoch in range(1):  # Single epoch
        # Randomly freeze a subset of layers
        layers_to_freeze = random.sample(layers, k=random.randint(1, len(layers) - 1))

        for layer in layers:
            requires_grad = layer not in layers_to_freeze
            for param in layer.parameters():
                param.requires_grad = requires_grad

        for idx, img_batch in enumerate(train_dataloader):

            img_batch = img_batch.to(device)  # Move to the device

            bbox_targets = bboxes_list[idx]  # bbox for the current image
            pred_target = pred_image[idx]

            optimizer.zero_grad()  # Reset gradients

            pred = model(img_batch)  # Make a prediction

            loss = criterion(pred, bbox_targets)  # Loss for boxes
            loss_outside = criterion_outside(pred, pred_target, bbox_targets) # loss for outside boxes


            total_loss = loss + loss_outside # total loss # Total loss
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Batch {idx + 1}, Loss: {loss.item()}")

    # Unfreeze all layers after training
    for param in model.parameters():
        param.requires_grad = True