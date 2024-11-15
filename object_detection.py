import cv2
import numpy as np


def create_object_mask(image, target_size=(128, 128)):  # Create a binary mask to identify possible obstacles in the image

    height, width, _ = image.shape
    cropped_image = image[:height // 2, :]
    resized_cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)  # Usa la sogliatura per identificare i tronchi

    return binary, resized_cropped_image


def find_object_areas(binary_mask, min_area_threshold=450):  # Find the areas in the binary mask > min_area_threshold
    inverted_mask = cv2.bitwise_not(binary_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
    valid_masks = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area_threshold:
            obj_mask = (labels == label).astype(np.uint8) * 255
            valid_masks.append(obj_mask)

    return valid_masks


def calculate_average_distance(depth_image, trunk_masks):  # Calculate the average distance from the camera based on the valid areas.
    distances = []

    for trunk_mask in trunk_masks:
        depth_values = depth_image[trunk_mask == 255]

        if depth_values.size > 0:
            average_distance = np.mean(depth_values)
            distances.append(average_distance)

    return min(distances)














