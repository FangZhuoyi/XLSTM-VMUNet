import os
import random
import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


class datasets(Dataset):
    """
    This class can handle both training and validation datasets and applies transformations as needed.
    """
    def __init__(self, path_Data, config, train=True):
        """
        Initialize the dataset by loading image and mask file paths.

        Args:
            path_Data (str): Path to the dataset directory.
            config: Configuration object containing the transformer functions for training and testing.
            train (bool): Flag to indicate if the dataset is for training or validation. Default is True (for training).
        """
        super().__init__()

        if train:
            # Load training image and mask file paths
            images_list = sorted(os.listdir(path_Data + 'train/images/'))
            masks_list = sorted(os.listdir(path_Data + 'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            # Set the transformer for training
            self.transformer = config.train_transformer
        else:
            # Load validation image and mask file paths
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            masks_list = sorted(os.listdir(path_Data + 'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            # Set the transformer for testing
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        """
        Retrieve a sample (image and mask) from the dataset by index.

        Args:
            indx (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding mask.
        """
        img_path, msk_path = self.data[indx]
        # Load image and mask, convert to numpy arrays
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255  # Normalize mask to [0, 1]
        # Apply transformation (augmentation, normalization, etc.)
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of image-mask pairs in the dataset.
        """
        return len(self.data)


def random_rot_flip(image, label):
    """
    Randomly rotate and flip an image and its corresponding mask.

    Args:
        image (np.ndarray): The image to be augmented.
        label (np.ndarray): The mask (label) to be augmented.

    Returns:
        tuple: The augmented image and label.
    """
    # Randomly rotate the image and label by 90 degrees (0, 1, 2, or 3 times)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    
    # Randomly select axis (0 for vertical, 1 for horizontal) for flipping
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()  # Perform flip
    label = np.flip(label, axis=axis).copy()  # Perform flip
    return image, label


def random_rotate(image, label):
    """
    Randomly rotate an image and its corresponding mask by a random angle.

    Args:
        image (np.ndarray): The image to be rotated.
        label (np.ndarray): The mask (label) to be rotated.

    Returns:
        tuple: The rotated image and label.
    """
    # Random angle between -20 and 20 degrees
    angle = np.random.randint(-20, 20)
    # Rotate both image and label by the same angle
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """
    Randomly apply rotations, flips, and resizing to an image and its corresponding mask.
    This class is intended to be used as a callable transformation for the dataset.
    """
    def __init__(self, output_size):
        """
        Initialize the random generator with the desired output size for the image and mask.

        Args:
            output_size (tuple): The desired output size (height, width) for the image and mask.
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        Apply random augmentations (rotations and flips), followed by resizing to the output size.

        Args:
            sample (dict): A dictionary containing 'image' and 'label' (mask).

        Returns:
            dict: A dictionary containing the transformed image and label.
        """
        image, label = sample['image'], sample['label']
        
        # Apply random rotations and flips with 50% probability for each transformation
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Resize image and label to the output size
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # Rescale the image using cubic interpolation
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # Cubic interpolation for image
            # Rescale the label using nearest-neighbor interpolation (for segmentation masks)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Convert the image to a tensor and add a channel dimension (for grayscale images)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # Add channel dimension
        # Convert the label (mask) to a tensor of type long (for categorical labels)
        label = torch.from_numpy(label.astype(np.float32))
        
        # Return the transformed image and label as a dictionary
        sample = {'image': image, 'label': label.long()}
        return sample
    