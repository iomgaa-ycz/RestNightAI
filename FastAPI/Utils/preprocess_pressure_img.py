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
import cv2


def extract_and_concat_blocks_corrected(image_np):
    """
    Extracts 240x240 blocks from specified positions in the image, considering a 2 pixel gap between each block,
    and concatenates them into a single numpy array.

    Parameters:
    - image_np: a numpy array of the image

    Returns:
    - concatenated_blocks: a numpy array containing the concatenated blocks
    """
    # Define the corrected coordinates for the blocks (left, upper, right, lower), considering the 2 pixel gap
    corrected_block_coords = [
        (0, 0, 240, 240),            # Block 1
        (242, 0, 242 + 240, 240),    # Block 2
        (484, 0, 484 + 240, 240),    # Block 3
        (726, 0, 726 + 240, 240),    # Block 4
        (0, 242, 240, 242 + 240),    # Block 5
        (242, 242, 242 + 240, 242 + 240),  # Block 6
        (484, 242, 484 + 240, 242 + 240),  # Block 7
        (726, 242, 726 + 240, 242 + 240)   # Block 8
    ]
    
    # Extract blocks
    blocks = [image_np[y1:y2, x1:x2] for (x1, y1, x2, y2) in corrected_block_coords]
    
    # Concatenate blocks horizontally and then vertically
    top_row_blocks = np.hstack(blocks[:4])
    bottom_row_blocks = np.hstack(blocks[4:])
    concatenated_blocks = np.vstack((top_row_blocks, bottom_row_blocks))
    
    return concatenated_blocks


def process_image(img_path, initial_rows_to_remove, initial_cols_to_remove, target_size=(160, 320)):
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
    # cropped_array = updated_enhanced_cropping_with_initials(img_path, initial_rows_to_remove, initial_cols_to_remove)
    cropped_array = extract_and_concat_blocks_corrected(img_path)
    
    
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

def base64_to_image_list(base64_list,folder,initial_rows,initial_cols,save_pic=False):
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
        begin_time = time.time()
        pressure_array = np.empty((len(base64_list), 160, 320))

        for i, base64_str in enumerate(base64_list):
            # 将Base64字符串转换为图像
            image_data = base64.b64decode(base64_str)
            # image = Image.open(BytesIO(image_data))

            # 将图像转换为三维数组
            # image_array = np.array(image)
            # 直接使用OpenCV从二进制数据读取图像
            image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            # 如果需要，将BGR格式转换为RGB格式
            temp_image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            # image_as_list = image_array.tolist()
            pressure_array[i,:,:] = process_image(temp_image_array, initial_rows, initial_cols)
        process_time = time.time() - begin_time
        print("Process time: ", process_time, "seconds")
    return pressure_array

def preprocess(data):
    data = np.array(data) / 255.0
    return data
    
