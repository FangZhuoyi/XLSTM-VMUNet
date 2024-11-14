from utils import *  
from datetime import datetime  
from torchvision import transforms  

class setting_config:
    # Network architecture being used
    network = 'XLSTM_VMUNet'
    
    # Configuration dictionary for the model
    model_config = {
        'num_classes': 1,  # Number of output classes (1 for binary segmentation)
        'input_channels': 3,  # Number of input channels (3 for RGB images)
        'depths': [2, 2, 2, 2],  # Depth of each encoder block (for U-Net style architecture)
        'depths_decoder': [2, 2, 2, 1],  # Depth for decoder blocks
        'drop_path_rate': 0.2,  # Drop path rate for regularization in deep architectures
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',  # Path to the pre-trained weights for initialization
    }
    
    # Data paths for input images and pre-trained weights
    data_path = './data/isic2018/'  # Path to the ISIC 2018 dataset
    pretrained_path = './pre_trained/'  # Path to store pre-trained weights
    
    # Loss function used for training (BCE + Dice Loss)
    criterion = BceDiceLoss(wb=1, wd=1)  # Binary Cross Entropy + Dice Loss with weight factors for each component
    
    # Dataset and training parameters
    datasets = 'isic2018'  # Name of the dataset being used
    num_classes = 1  # Number of output classes (binary segmentation)
    
    # Image dimensions for resizing
    input_size_h = 256  # Height of input image
    input_size_w = 256  # Width of input image
    input_channels = 3  # RGB channels
    
    # Distributed training flags (disabled by default)
    distributed = False  # Whether to use distributed training
    local_rank = -1  # Local rank for multi-GPU setup (set to -1 for single GPU)
    
    # Number of worker threads for loading data
    num_workers = 0  # Number of workers to load data (0 for default)
    
    # Random seed for reproducibility
    seed = 42  # Seed for random number generators (for reproducibility)
    
    # Multi-GPU setup parameters (if using distributed training)
    world_size = None  # Total number of processes (used in distributed training)
    rank = None  # The rank of the current process (used in distributed training)
    
    # Mixed Precision Training Flag (whether to use AMP for faster training on supported hardware)
    amp = False  # If True, use automatic mixed precision
    
    # GPU ID for single-GPU training (or multi-GPU setup)
    gpu_id = '0'  # Use the first GPU for training
    
    # Hyperparameters for batch size and number of epochs
    batch_size = 8  # Number of samples per batch
    epochs = 150  # Number of epochs for training
    
    # Training, validation, and checkpointing intervals
    print_interval = 20  # How often to print progress (every 20 iterations)
    val_interval = 1  # Validation interval (every epoch)
    save_interval = 5  # Interval for saving model checkpoints (every 5 epochs)
    
    # Threshold for binary classification (used in segmentation)
    threshold = 0.5  # Threshold for binarizing segmentation outputs (e.g., 0.5 for binary segmentation)
    
    # Directory for saving experiment results, dynamically generated based on current date and time
    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
    
    # Data transformations for training and testing
    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),  # Normalize input images based on dataset statistics
        myToTensor(),  # Convert images to PyTorch tensor format
        myRandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
        myRandomVerticalFlip(p=0.5),  # Random vertical flip with 50% probability
        myRandomRotation(p=0.5, degree=[0, 360]),  # Random rotation between 0 and 360 degrees
        myResize(input_size_h, input_size_w)  # Resize images to a fixed size (256x256)
    ])
    
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),  # Normalize test images
        myToTensor(),  # Convert to tensor format
        myResize(input_size_h, input_size_w)  # Resize images to fixed size for testing
    ])
    
    # Optimizer choice (AdamW by default)
    opt = 'AdamW'  # Optimizer choice, can be changed to others like Adam, SGD, etc.
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    
    # Optimizer hyperparameters based on selected optimizer
    if opt == 'Adadelta':
        lr = 0.01  # Learning rate for Adadelta
        rho = 0.9  # Coefficient for computing running averages of squared gradients
        eps = 1e-6  # Small constant to prevent division by zero
        weight_decay = 0.05  # L2 penalty
    elif opt == 'Adagrad':
        lr = 0.01  # Learning rate for Adagrad
        lr_decay = 0  # Learning rate decay
        eps = 1e-10  # Small constant to improve numerical stability
        weight_decay = 0.05  # L2 penalty
    elif opt == 'Adam':
        lr = 0.001  # Learning rate for Adam
        betas = (0.9, 0.999)  # Coefficients for computing running averages of gradients
        eps = 1e-8  # Small constant for numerical stability
        weight_decay = 0.0001  # L2 penalty
        amsgrad = False  # Whether to use AMSGrad variant of Adam
    elif opt == 'AdamW':
        lr = 0.001  # Learning rate for AdamW
        betas = (0.9, 0.999)  # Coefficients for Adam
        eps = 1e-8  # Small constant for numerical stability
        weight_decay = 1e-2  # Weight decay for regularization
        amsgrad = False  # Use standard AdamW
    elif opt == 'Adamax':
        lr = 2e-3  # Learning rate for Adamax
        betas = (0.9, 0.999)  # Coefficients for Adamax
        eps = 1e-8  # Small constant for numerical stability
        weight_decay = 0  # No weight decay
    elif opt == 'SGD':
        lr = 0.01  # Learning rate for SGD
        momentum = 0.9  # Momentum factor for SGD
        weight_decay = 0.05  # L2 penalty
        dampening = 0  # Dampening for momentum
        nesterov = False  # Whether to use Nesterov momentum

    # Learning rate scheduler choice
    sch = 'CosineAnnealingLR'  # Scheduler type, can be changed to others like 'StepLR', 'ExponentialLR', etc.
    
    if sch == 'CosineAnnealingLR':
        T_max = 50  # Number of iterations for the cosine annealing period
        eta_min = 0.00001  # Minimum learning rate
        last_epoch = -1  # Last epoch for learning rate scheduler
    
    # Other possible schedulers: 'StepLR', 'MultiStepLR', 'ReduceLROnPlateau', etc.
    # Each scheduler has specific parameters that control how the learning rate decreases
