import os
import sys
import torch
import warnings
from engine import *
from utils import *
from config_setting import setting_config
from dataset import datasets
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from XLSTM_VMUNet import XLSTM_VMUNet

# Suppress all warnings to avoid clutter in the console
warnings.filterwarnings("ignore")


def main(config):
    # Initialize logger and directory structure for saving models and outputs
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')  # Log directory for training progress
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')  # Checkpoint directory for model saves
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')  # Path to resume model if it exists
    outputs = os.path.join(config.work_dir, 'outputs')  # Directory for saving output results

    # Create directories if they do not exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    # Set up the logger for training logs
    global logger
    logger = get_logger('train', log_dir)
    # Set up TensorBoard writer for logging metrics
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    # Log configuration details for reference
    log_config_info(config, logger)

    # Initialize GPU settings
    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id  # Set which GPU(s) to use
    set_seed(config.seed)  # Set the random seed for reproducibility
    torch.cuda.empty_cache()  # Clear GPU cache to avoid memory issues

    # Load training and validation datasets
    print('#----------Preparing dataset----------#')
    train_dataset = datasets(config.data_path, config, train=True)  # Training dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)  # DataLoader for training

    val_dataset = datasets(config.data_path, config, train=False)  # Validation dataset
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)  # DataLoader for validation

    # Initialize the model with configuration parameters
    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config  # Get model config from the configuration file
    model = XLSTM_VMUNet(
        num_classes=model_cfg['num_classes'],  # Number of output classes
        input_channels=model_cfg['input_channels'],  # Number of input channels
        depths=model_cfg['depths'],  # Depths of encoder layers
        depths_decoder=model_cfg['depths_decoder'],  # Depths of decoder layers
        drop_path_rate=model_cfg['drop_path_rate'],  # Drop path rate for regularization
        load_ckpt_path=model_cfg['load_ckpt_path'],  # Pretrained checkpoint path (if any)
    )

    # Load pretrained weights (if available)
    model.load_from()
    model = model.cuda()  # Move the model to GPU
    # Calculate parameters and FLOPS for the model (useful for model analysis)
    cal_params_flops(model, 256, logger)

    # Set up the loss function, optimizer, and learning rate scheduler
    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion  # Loss function
    optimizer = get_optimizer(config, model)  # Optimizer (e.g., Adam, SGD)
    scheduler = get_scheduler(config, optimizer)  # Learning rate scheduler

    # Set initial values for tracking best model performance
    print('#----------Set other params----------#')
    min_loss = 999  # Initialize the minimum loss to a large value
    start_epoch = 1  # Default start epoch is 1
    min_epoch = 1  # Epoch with minimum loss
    if os.path.exists(resume_model):  # Check if a checkpoint to resume exists
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))  # Load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
        saved_epoch = checkpoint['epoch']  # Get saved epoch
        start_epoch += saved_epoch  # Resume from the next epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)  # Log the information of resuming from checkpoint

    step = 0  # Initialize training step counter
    print('#----------Training----------#')
    # Training loop: run for the specified number of epochs
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()  # Empty GPU cache before each epoch
        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )  # Train for one epoch

        # Validate the model after each epoch
        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        # Save the model if the validation loss has improved
        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))  # Save best model
            min_loss = loss  # Update minimum loss
            min_epoch = epoch  # Update epoch with the best performance

        # Save the model and other state information after each epoch
        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))  # Save the latest checkpoint

    # After training, test the best model on validation data
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)  # Load the best model weights
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )  # Test the model on the validation dataset

        # Rename the best model checkpoint to include epoch and loss information
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    # Load the configuration settings and start training
    config = setting_config
    main(config)

