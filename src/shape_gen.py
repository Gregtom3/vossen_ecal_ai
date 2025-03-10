import numpy as np
import tensorflow as tf
from skimage.draw import polygon, disk
import random
from copy import deepcopy
import matplotlib.pyplot as plt

def is_dark(color):
    # Ensure the color is a numpy array
    color = np.array(color)
    # Calculate luminance
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    # Define the threshold for darkness
    threshold = 0.15
    # Determine if the color is dark
    return luminance < threshold

def generate_blank_image(image_size):
    image = np.ones((*image_size, 5))  # Shape = [image_width, image_height, 5]
    image_width, image_height = image_size
    for col in range(image_width):
        image[:, col, 4] = col / image_width
    for row in range(image_height):
        image[row, :, 3] = row / image_height
    return image

def draw_triangle(image, center, size):
    r, c = center
    points = np.array([[r - size, c],
                       [r + size, c - size],
                       [r + size, c + size]])
    rr, cc = polygon(points[:, 0], points[:, 1], image.shape)
    return rr, cc

def draw_square(image, center, size):
    r, c = center
    half_size = size // 2
    rr, cc = polygon([r - half_size, r + half_size, r + half_size, r - half_size],
                     [c - half_size, c - half_size, c + half_size, c + half_size],
                     image.shape)
    return rr, cc

def draw_circle(image, center, size):
    rr, cc = disk(center, size, shape=image.shape)
    return rr, cc

def generate_dataset(
    num_images,
    image_size=(64, 64),
    min_shapes=1,
    max_shapes=9,
    shape_size_range=(8, 10),
    same_color=False,
    same_shape=None,
    shape_overlap_max=None  # Percentage overlap allowed; discard image if exceeded
):
    images = []
    shape_types = ['circle', 'triangle', 'square'] if same_shape is None else [same_shape]
    unique_shape_id = 1

    for _ in range(num_images):
        valid_image = False
        while not valid_image:
            # Generate a blank image and output container
            image = generate_blank_image(image_size)
            output_image = np.zeros((*image_size, 7), dtype=np.float32)
            
            num_shapes = random.randint(min_shapes, max_shapes)
            shape_masks = []  # Store individual shape masks
            
            for _ in range(num_shapes):
                shape_type = random.choice(shape_types)
                size = random.randint(*shape_size_range)
                center = (np.random.rand() * image_size[0], np.random.rand() * image_size[1])
                
                if shape_type == 'triangle':
                    rr, cc = draw_triangle(image, center, size)
                    shape_type_id = 2
                elif shape_type == 'square':
                    rr, cc = draw_square(image, center, size)
                    shape_type_id = 3
                elif shape_type == 'circle':
                    rr, cc = draw_circle(image, center, size)
                    shape_type_id = 1

                # Ensure valid coordinates within bounds
                rr = np.clip(rr, 0, image_size[0] - 1)
                cc = np.clip(cc, 0, image_size[1] - 1)

                # Assign random RGB colors to the shape and update shape info
                if same_color:
                    image[rr, cc, 0:3] = np.array([1, 0, 0])
                else:
                    image[rr, cc, 0:3] = np.random.rand(3)
                output_image[rr, cc, 5] = unique_shape_id
                output_image[rr, cc, 6] = shape_type_id

                # Create and store the shape mask
                mask = np.zeros(image_size, dtype=bool)
                mask[rr, cc] = True
                shape_masks.append(mask)
                unique_shape_id += 1
            
            # Check for overlap if shape_overlap_max is specified
            if shape_overlap_max is not None:
                overlap_area = np.sum(np.logical_and.reduce(shape_masks))
                total_area = np.sum(np.logical_or.reduce(shape_masks))
                overlap_percentage = (overlap_area / total_area) * 100

                if overlap_percentage > shape_overlap_max:
                    continue  # Discard the image and regenerate

            # Copy the RGB image to the first 3 channels of the output image
            output_image[..., :5] = image

            # Check if the output_image contains at least one shape
            valid_image = np.any(output_image[..., 5] > 0)
        images.append(output_image)
    
    # Return a TensorFlow tensor
    return tf.convert_to_tensor(np.array(images))
