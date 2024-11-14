import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import save_imgs

# Function for training the model for one epoch
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    model.train()  # Set model to training mode
    loss_list = []  # List to track the loss for each iteration
    
    # Iterate through each batch of data in the training data loader
    for iter, data in enumerate(train_loader):
        step += iter  # Increment global step
        optimizer.zero_grad()  # Clear previous gradients
        
        # Extract images and targets (ground truth masks)
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()  # Move to GPU
        
        # Perform forward pass
        out = model(images)
        
        # Calculate the loss between predicted outputs and ground truth
        loss = criterion(out, targets)
        
        # Perform backward pass (compute gradients)
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Append current loss to the loss list
        loss_list.append(loss.item())
        
        # Log current learning rate
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        # Write loss to tensorboard (or other writer)
        writer.add_scalar('loss', loss, global_step=step)
        
        # Print and log information periodically
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)  # Print log to console
            logger.info(log_info)  # Log to file
            
    # Update the learning rate scheduler after each epoch
    scheduler.step()
    
    return step  # Return the updated global step

# Function for evaluating the model on the validation set for one epoch
def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    model.eval()  # Set model to evaluation mode
    preds = []  # List to store model predictions
    gts = []  # List to store ground truth labels
    loss_list = []  # List to track loss
    
    # Disable gradient calculation for validation to save memory
    with torch.no_grad():
        # Iterate through the validation data loader
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()  # Move to GPU
            
            # Perform forward pass
            out = model(img)
            
            # Calculate the loss
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            
            # Append ground truth and predictions (squeezed to remove unnecessary dimensions)
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:  # In case the model outputs a tuple, take the first element
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)  # Append prediction
            
    # Only log and calculate metrics for specific validation intervals
    if epoch % config.val_interval == 0:
        # Flatten predictions and ground truth arrays
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        
        # Convert predictions and ground truth to binary (thresholding)
        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)
        
        # Compute confusion matrix
        confusion = confusion_matrix(y_true, y_pre)
        
        # Extract metrics from the confusion matrix
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1] 
        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        
        # Log and print metrics
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                    specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)  # Print log to console
        logger.info(log_info)  # Log to file
    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)  # Return the average loss for the epoch

# Function for testing the model on the test set
def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    model.eval()  # Set model to evaluation mode
    preds = []  # List to store model predictions
    gts = []  # List to store ground truth labels
    loss_list = []  # List to track loss
    
    # Disable gradient calculation for testing to save memory
    with torch.no_grad():
        # Iterate through the test data loader
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()  # Move to GPU
            
            # Perform forward pass
            out = model(img)
            
            # Calculate the loss
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            
            # Append ground truth and predictions (squeezed to remove unnecessary dimensions)
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:  # In case the model outputs a tuple, take the first element
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)  # Append prediction
            
            # Save images at specific intervals
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
        
        # Flatten predictions and ground truth arrays
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        
        # Convert predictions and ground truth to binary (thresholding)
        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)
        
        # Compute confusion matrix
        confusion = confusion_matrix(y_true, y_pre)
        
        # Extract metrics from the confusion matrix
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1] 
        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        
        # Log the test results
        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                    specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)  # Print log to console
        logger.info(log_info)  # Log to file
    
    return np.mean(loss_list)  # Return the average loss for the epoch
