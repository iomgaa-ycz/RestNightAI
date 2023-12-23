import numpy as np
from PIL import Image
import pandas as pd
import json
import time
import os
import base64
from PIL import Image
from io import BytesIO
import numpy as np

def enhanced_cropping(img_array):
    """
    Enhanced cropping method following the new requirements.
    """
    # Load the image
    # img = Image.open(img_path)
    # img_array = np.array(img)

    # Initial removal of fully transparent rows and columns
    img_array = np.array(img_array)
    rows_fully_transparent = np.all(img_array[:, :, 3] == 0, axis=1)
    cols_fully_transparent = np.all(img_array[:, :, 3] == 0, axis=0)
    
    # Remove rows and columns where the proportion of fully transparent pixels is above 80%
    rows_high_transparency = np.sum(img_array[:, :, 3] == 0, axis=1) / img_array.shape[1] > 0.8
    cols_high_transparency = np.sum(img_array[:, :, 3] == 0, axis=0) / img_array.shape[0] > 0.8

    # Combine the conditions
    rows_to_remove = np.logical_or(rows_fully_transparent, rows_high_transparency)
    cols_to_remove = np.logical_or(cols_fully_transparent, cols_high_transparency)

    # Crop the image
    cropped_array = img_array[~rows_to_remove][:, ~cols_to_remove]
    
    return cropped_array

def process_image(img_path, target_size=(160, 320)):
    """
    Process the image by cropping, downsampling, and mapping RGB values.
    
    Args:
    - img_path (str): The path to the input image.
    - mapping (dict): A dictionary mapping RGB tuples to values.
    - target_size (tuple): The target size for downsampling (default is (160, 320)).
    
    Returns:
    - numpy.ndarray: The processed image array.
    """

    # Enhanced cropping
    cropped_array = enhanced_cropping(img_path)
    
    # Downsampling
    downsampled_array = downsample_image_custom(cropped_array, target_size)
    
    # Mapping RGB values
    mapped_array = map_rgb_to_values(downsampled_array)
    
    return mapped_array.tolist()

def map_rgb_to_values(img_array):
    """
    Map the RGB values in an image array to their corresponding values using a provided mapping.
    
    Args:
    - img_array (numpy.ndarray): The input image array of shape (height, width, channels).
    - mapping (dict): A dictionary mapping RGB tuples to values.
    
    Returns:
    - numpy.ndarray: A new array of shape (height, width, 1) with the mapped values.
    """
    img_array = img_array[:, :, :3]
    img_array = (img_array[:,:,0]+img_array[:,:,1])/2
    #向上取整
    img_array = np.ceil(img_array)
    return img_array

def downsample_image_custom(img_array, target_size=(160, 320)):
    # Calculate the downsampling rate for height and width
    downsample_rate_height = img_array.shape[0] / target_size[0]
    downsample_rate_width = img_array.shape[1] / target_size[1]
    
    # Extract rows and columns based on the downsampling rate
    selected_rows = np.arange(0, img_array.shape[0], downsample_rate_height).astype(int)
    selected_cols = np.arange(0, img_array.shape[1], downsample_rate_width).astype(int)
    
    # Use numpy's advanced indexing to extract the desired rows and columns
    downsampled_array = img_array[selected_rows][:, selected_cols]
    
    return downsampled_array

def base64_to_image_list(base64_list,folder,save_pic=False):
    if save_pic:
        if not os.path.exists('./pic'):
            os.makedirs('./pic')  # 创建文件夹，如果不存在

        if not os.path.exists(f'./pic/{folder}'):
            os.makedirs(f'./pic/{folder}') # 创建文件夹，如果不存在

        image_list = []

        for i, base64_str in enumerate(base64_list):
            # 将Base64字符串转换为图像
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))

            # 保存图像
            image_path = f'./pic/{folder}/image_{i}.png'
            image.save(image_path)

            # 将图像转换为三维数组
            image_array = np.array(image)
            image_as_list = image_array.tolist()
            image_list.append(process_image(image_as_list))
    else:
        image_list = []

        for i, base64_str in enumerate(base64_list):
            # 将Base64字符串转换为图像
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))

            # 将图像转换为三维数组
            image_array = np.array(image)
            image_as_list = image_array.tolist()
            image_list.append(process_image(image_as_list))
    return image_list

def preprocess(data):
    data = np.array(data) / 255.0
    
