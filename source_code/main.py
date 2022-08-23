from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.utils import save_image

# --- PARAMETERS ---

# Image names
input_image_name = 'input.jpg'
output_image_name = 'output.png'

# Gaussian Blur
kernel_length = 15
sigma_value = 9

# --- MAIN ---

if __name__ == '__main__':
    # Read image
    orig_img = Image.open(input_image_name)
    # Compose transforms
    transforms = T.Compose([
        T.ToTensor() ,
        T.Grayscale(1) ,
        T.Normalize(0, 1, inplace=False) ,
        T.GaussianBlur(kernel_size=(kernel_length, kernel_length), sigma=(sigma_value, sigma_value))
    ])
    # Apply transforms
    tensor_img = transforms(orig_img)
    # Save output image
    save_image(tensor_img, output_image_name)
