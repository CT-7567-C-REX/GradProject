from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch import nn

class eval_datasets(Dataset):
    def __init__(self, directory):
        self.directory = directory
        
        self.img_files = [img_file for img_file in os.listdir(directory) if (img_file.endswith('Plan.jpg'))]

        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        
        selected_img_file = self.img_files[index]
        selected_mask_file = selected_img_file.replace("Plan.jpg", "Seg.jpg")
        
        
        #convert the plan and mask from their path to PIL image. 
        plan = Image.open(os.path.join(self.directory, selected_img_file)) #RGB mode
        plan_ori = Image.open(os.path.join(self.directory, selected_img_file)).convert('1')
        mask = Image.open(os.path.join(self.directory, selected_mask_file)).convert('L')
        
        plan = np.array(plan).astype(np.float32)
        plan_ori = np.array(plan_ori).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        # Add dummy channel dimension
        mask = mask[..., np.newaxis]
        plan_ori = plan_ori[..., np.newaxis]
        
        mask_original = mask.copy()
        mask = np.zeros_like(mask).astype(np.float32)
         
        # SELECT MASKS ##
        ################
        # [  0.  29. 76. 150. 255.]
        mask[(mask_original <= 25.0)] = 0.0
        mask[(mask_original >= 26.0) & (mask_original <= 120.0)] = 1.0
        mask[(mask_original >= 121.0) & (mask_original <= 200.0)] = 2.0
        mask[(mask_original >= 201.0) & (mask_original <= 230.0)] = 3.0
        mask[(mask_original >= 231.0)] = 4.0
        
        # convert to tensor
        # (Width, Height, Channel) -> (Channel, Width, Height)
        plan = torch.from_numpy(plan.copy().transpose((2,0,1))) 
        mask = torch.from_numpy(mask.copy().transpose((2,0,1)))
        
        return plan, mask 

def calculate_miou(predictions, targets, color_mapping):
    iou_per_class = []

    for label in color_mapping.keys():
        iou = calculate_iou(predictions,targets, label)
        iou_per_class.append(iou)

    miou = (sum(iou_per_class) / len(iou_per_class)) * 100.0

    return miou

def calculate_iou(prediction, target, class_label):

    prediction = torch.round(prediction).type(torch.LongTensor).cpu()
    target = target.cpu()
    
    prediction_mask = (prediction == class_label)
    target_mask = (target == class_label)

    intersection = torch.logical_and(prediction_mask, target_mask)
    union = torch.logical_or(prediction_mask, target_mask)

    if union.sum() == 0:
        iou = 1.0
    else:
        iou = intersection.sum() / union.sum()

    return iou

@torch.no_grad()
def eval_fn(model, dataloader, device):
    
    criterion = nn.MSELoss()
    total_correct = 0
    total_pixels = 0
    avgloss = []
    miou_values = []
    compute_avg = lambda x: sum(x) / len(x) if x else 0
    color_mapping = {
        0: (0, 0, 0),       # Walls
        1: (80, 80, 255),   # Room
        2: (255, 80, 80),   # Iwan
        3: (255, 255, 0),   # Stairs
        4: (255, 255, 255), # Background  
    }
    
    for plan, mask in dataloader:
        plan = plan.to(device)
        mask = mask.to(device)
        
       # Forward pass with mixed precision
        with torch.autocast(device_type=device, dtype=torch.float16):
            # Get model prediction
            pred_mask = model(plan)  # Shape: [batch_size, 1, 512, 512]
            loss = criterion(pred_mask, mask)
            avgloss.append(loss.item())
            
            # Round the predictions and calculate correct pixels
            pred_mask = torch.round(pred_mask)
            total_correct += (pred_mask == mask).sum().item()
            total_pixels += torch.numel(pred_mask)

            # Compute mIoU for this batch
            batch_miou = calculate_miou(pred_mask, mask, color_mapping)
            miou_values.append(batch_miou)

    avg_acc = (total_correct / total_pixels) * 100
    return compute_avg(avgloss), compute_avg(miou_values), avg_acc
