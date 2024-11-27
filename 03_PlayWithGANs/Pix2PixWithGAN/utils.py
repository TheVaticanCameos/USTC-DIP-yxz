import os
import cv2
import torch
import numpy as np


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().detach().numpy()
    if image.shape[0] != 3:
        raise ValueError(f"Expected tensor with 3 channels, but got {image.shape[0]}.")
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = image[..., ::-1]
    return image


def save_images(inputs, targets, outputs, folder_name: str, epoch: int, num_images: int = 5) -> None:
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        input_img = tensor_to_image(inputs[i])
        target_img = tensor_to_image(targets[i])
        output_img = tensor_to_image(outputs[i])
        comparison = np.hstack((input_img, target_img, output_img))
        comparison_path = f'{folder_name}/epoch_{epoch}/result_{i + 1}.png'
        cv2.imwrite(comparison_path, comparison)
