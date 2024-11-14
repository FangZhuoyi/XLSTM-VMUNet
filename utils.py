import logging
import logging.handlers
import math
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
from thop import profile
from matplotlib import pyplot as plt


# Function to set random seed for reproducibility
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set the environment variable to control hash randomization
    random.seed(seed)  # Set Python's random seed
    np.random.seed(seed)  # Set numpy's random seed
    torch.manual_seed(seed)  # Set PyTorch's seed for CPU
    torch.cuda.manual_seed(seed)  # Set PyTorch's seed for the current GPU
    torch.cuda.manual_seed_all(seed)  # Set PyTorch's seed for all GPUs
    cudnn.benchmark = False  # Disable benchmarking for deterministic results
    cudnn.deterministic = True  # Ensure deterministic algorithms are used in CUDA operations


# Function to configure and return a logger
def get_logger(name, log_dir):
    if not os.path.exists(log_dir):  # Check if the log directory exists
        os.makedirs(log_dir)  # Create the log directory if it doesn't exist
    logger = logging.getLogger(name)  # Create a logger object
    logger.setLevel(logging.INFO)  # Set the logging level to INFO
    info_name = os.path.join(log_dir, '{}.info.log'.format(name))  # Define the log file path
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',  # Log file rotates daily
                                                             encoding='utf-8')  # Set file encoding to UTF-8
    info_handler.setLevel(logging.INFO)  # Set the handler's log level
    formatter = logging.Formatter('%(asctime)s - %(message)s',  # Log format includes time and message
                                  datefmt='%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(formatter)  # Set the formatter for the handler
    logger.addHandler(info_handler)  # Add the handler to the logger
    return logger


# Function to log the configuration information (hyperparameters, settings)
def log_config_info(config, logger):
    config_dict = config.__dict__  # Convert the config object to a dictionary
    log_info = f'#----------Config info----------#'  # Print a header
    logger.info(log_info)  # Log the header
    for k, v in config_dict.items():  # Iterate over the dictionary to log all key-value pairs
        if k[0] == '_':  # Skip private attributes
            continue
        else:
            log_info = f'{k}: {v},'  # Format the log info
            logger.info(log_info)  # Log the key-value pair


# Function to return an optimizer based on the configuration
def get_optimizer(config, model):
    # Check if the chosen optimizer is supported
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'
    # Create the corresponding optimizer based on the configuration
    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


# Function to return a learning rate scheduler based on the configuration
def get_scheduler(config, optimizer):
    # Check if the chosen scheduler is supported
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    # Create the corresponding scheduler based on the configuration
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        # Custom scheduler for warm-up and multi-step LR decay
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        # Custom scheduler for warm-up and cosine LR decay
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler


# Function to save the input image, ground truth mask, and predicted mask
def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Convert tensor to numpy array
    img = img / 255. if img.max() > 1.1 else img  # Normalize the image if necessary
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)  # Convert to binary mask
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)  # Threshold the prediction

    # Create subplots to display the images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[1].imshow(msk)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(msk_pred)
    ax[2].set_title("Prediction")
    
    # Save the figure as an image file
    save_name = os.path.join(save_path, f'{test_data_name}_{i}.png')
    plt.savefig(save_name)
    plt.close(fig)  # Close the figure to free up memory
    print(f"Saved {save_name}")
    
    
# Binary Cross-Entropy (BCE) Loss Class
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        # Initialize the standard BCE loss
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        # Flatten the input prediction and target tensors
        size = pred.size(0)  # Get the batch size
        pred_ = pred.view(size, -1)  # Flatten prediction tensor
        target_ = target.view(size, -1)  # Flatten target tensor

        # Calculate and return BCE loss
        return self.bceloss(pred_, target_)


# Dice Loss Class (used for segmentation tasks)
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        # Define a smoothing constant to prevent division by zero
        smooth = 1
        size = pred.size(0)

        # Flatten the prediction and target tensors
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        # Calculate the intersection between prediction and target
        intersection = pred_ * target_

        # Calculate the Dice score
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)

        # Calculate the Dice loss (1 - Dice score)
        dice_loss = 1 - dice_score.sum() / size
        return dice_loss


# Combined BCE and Dice Loss (for binary segmentation)
class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()  # Binary Cross-Entropy loss
        self.dice = DiceLoss()  # Dice loss
        self.wb = wb  # Weight for BCE loss
        self.wd = wd  # Weight for Dice loss

    def forward(self, pred, target):
        # Calculate BCE loss
        bceloss = self.bce(pred, target)

        # Calculate Dice loss
        diceloss = self.dice(pred, target)

        # Combine both losses with the specified weights
        loss = self.wd * diceloss + self.wb * bceloss
        return loss


# Data transformation: Convert data to PyTorch tensors
class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        # Convert the image and mask to tensor format and rearrange dimensions to CHW
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


# Data transformation: Resize image and mask to a given size
class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h  # Height
        self.size_w = size_w  # Width

    def __call__(self, data):
        image, mask = data
        # Resize both the image and mask
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


# Data transformation: Random horizontal flip
class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p  # Probability of flipping

    def __call__(self, data):
        image, mask = data
        # Flip the image and mask horizontally with probability p
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


# Data transformation: Random vertical flip
class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p  # Probability of flipping

    def __call__(self, data):
        image, mask = data
        # Flip the image and mask vertically with probability p
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


# Data transformation: Random rotation
class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])  # Random angle within the specified range
        self.p = p  # Probability of rotation

    def __call__(self, data):
        image, mask = data
        # Rotate the image and mask with a random angle with probability p
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask


# Data transformation: Normalize images based on dataset statistics
class myNormalize:
    def __init__(self, data_name, train=True):
        self.mean = 157.561 if train else 149.034
        self.std = 26.706 if train else 32.022

    def __call__(self, data):
        img, msk = data
        # Normalize the image using the mean and std values
        img_normalized = (img - self.mean) / self.std
        # Rescale to the 0-255 range after normalization
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk


def cal_params_flops(model, size, logger):

    # Generate a random tensor representing an input image of shape (1, 3, size, size)
    input = torch.randn(1, 3, size, size).cuda()

    # Use the profile function from torchprofile to compute FLOPs and parameters
    flops, params = profile(model, inputs=(input,))

    # Print the FLOPs and parameters in more human-readable units (GFLOPs and M)
    print('flops', flops / 1e9)  # FLOPs in GFLOPs
    print('params', params / 1e6)  # Parameters in M (Mega parameters)

    # Compute the total number of parameters in the model manually
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))  # Total parameters in Mega parameters

    # Log the FLOPs, parameters, and total parameters
    logger.info(f'flops: {flops / 1e9}, params: {params / 1e6}, Total params: {total / 1e6:.4f}')

def calculate_metric_percase(pred, gt):

    # Convert both the predicted and ground truth masks to binary (values > 0 become 1)
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    # If both the prediction and ground truth contain positive pixels, compute the metrics
    if pred.sum() > 0 and gt.sum() > 0:
        # Calculate the Dice coefficient and Hausdorff distance (95th percentile)
        dice = metric.binary.dc(pred, gt)  # Dice coefficient
        hd95 = metric.binary.hd95(pred, gt)  # HD95 distance
        return dice, hd95
    # If the prediction has positive pixels but the ground truth doesn't, return perfect Dice and HD95 = 0
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    # If there are no positive pixels in either, return 0 for both metrics
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], 
                       test_save_path=None, case=None, z_spacing=1, val_or_test=False):

    # Squeeze the input to remove extra dimensions, then convert to numpy arrays
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # Initialize the prediction array to hold results for each slice
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        # Process each slice of the 3D image independently
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            
            # Resize the slice if its size doesn't match the target patch size
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # interpolation with cubic order (3)

            # Convert the slice to a tensor and add batch and channel dimensions
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            # Set the network to evaluation mode and disable gradient computation for inference
            net.eval()
            with torch.no_grad():
                # Forward pass through the model
                outputs = net(input)
                
                # Apply softmax to get probabilities and take the class with the highest probability
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                
                # Convert the prediction back to a numpy array
                out = out.cpu().detach().numpy()

                # If resizing was performed, undo the resizing of the predicted output
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out

                # Store the prediction for this slice
                prediction[ind] = pred
    else:
        # If the input is not 3D, process the whole image at once
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            # Forward pass through the model for 2D images
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    # Calculate metrics (Dice and HD95) for each class (excluding background)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # If saving images is required, save the input, predicted, and ground truth images as NIfTI files
    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        
        # Set the spacing for each image (z-spacing)
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        
        # Save the images as NIfTI files (.nii.gz format)
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")

    # Return the computed metrics for all classes
    return metric_list
