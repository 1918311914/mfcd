import numpy as np
import PIL.Image
import torch
import torch.fft as fft
import math


def ideal_high_pass_filter(
        image: PIL.Image.Image,
        cutoff: float,
        device: torch.device = torch.device('cpu')
) -> PIL.Image.Image:
    """
    Apply a high-pass filter to the input image.

    Args:
        image (PIL.Image.Image): The input image to be filtered.
        cutoff (int): The cutoff frequency for the high-pass filter.
        device (torch.device): The device to run the computation on. Default is CPU.

    Returns:
        PIL.Image.Image: The filtered image after applying the high-pass filter.
    """
    # Convert the image to a PyTorch tensor and move it to the specified device
    if cutoff > 1.0:
        cutoff = int(cutoff)
    else:
        cutoff = int(math.sqrt((image.size[0] / 2) ** 2 + (image.size[1] / 2) ** 2) * cutoff)
    if len(image.size) > 2:
        image = image.convert("RGB")
    img_tensor = torch.from_numpy(np.array(image)).float().to(device)
    if len(img_tensor.shape) == 2:  # Grayscale image
        filtered_tensor = _apply_ideal_high_pass(img_tensor, cutoff, device)
    else:  # Color image
        filtered_channels = []
        for channel in range(img_tensor.shape[2]):
            filtered_channel = _apply_ideal_high_pass(img_tensor[:, :, channel], cutoff, device)
            filtered_channels.append(filtered_channel)
        filtered_tensor = torch.stack(filtered_channels, dim=2)

    # Move the tensor back to CPU before converting to a PIL image
    filtered_tensor = filtered_tensor.cpu()
    filtered_image = PIL.Image.fromarray(filtered_tensor.numpy().astype(np.uint8))

    del img_tensor, filtered_tensor
    # Free up GPU memory if the device is CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return filtered_image


def _apply_ideal_high_pass(
        channel: torch.Tensor,
        cutoff: int,
        device: torch.device
) -> torch.Tensor:
    """
    Apply a high-pass filter to a single channel of an image.

    Args:
        channel (torch.Tensor): A single channel of the image.
        cutoff (int): The cutoff frequency for the high-pass filter.
        device (torch.device): The device to run the computation on.

    Returns:
        torch.Tensor: The filtered channel after applying the high-pass filter.
    """
    # Perform a 2D Fourier transform
    f = fft.fft2(channel)
    fshift = fft.fftshift(f)

    # Create a high-pass filter
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
    distance = torch.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    mask = torch.ones((rows, cols), dtype=torch.float32, device=device)
    mask[distance <= cutoff] = 0

    # Apply the filter
    fshift = fshift * mask

    # Perform an inverse Fourier transform
    f_ishift = fft.ifftshift(fshift)
    img_back = fft.ifft2(f_ishift)
    img_back = torch.abs(img_back)

    # Normalize to the range of 0 - 255
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255
    del fshift, f_ishift, distance, mask
    return img_back


def ideal_low_pass_filter(
        image: PIL.Image.Image,
        cutoff: float,
        device: torch.device = torch.device('cpu')
) -> PIL.Image.Image:
    """
    Apply a low-pass filter to the input image.

    Args:
        image (PIL.Image.Image): The input image to be filtered.
        cutoff (int): The cutoff frequency for the low-pass filter.
        device (torch.device): The device to run the computation on. Default is CPU.

    Returns:
        PIL.Image.Image: The filtered image after applying the low-pass filter.
    """
    # Convert the image to a PyTorch tensor and move it to the specified device
    if cutoff > 1.0:
        cutoff = int(cutoff)
    else:
        cutoff = int(math.sqrt((image.size[0] / 2) ** 2 + (image.size[1] / 2) ** 2) * cutoff)
    if len(image.size) > 2:
        image = image.convert("RGB")
    img_tensor = torch.from_numpy(np.array(image)).float().to(device)
    if len(img_tensor.shape) == 2:  # Grayscale image
        filtered_tensor = _apply_ideal_low_pass(img_tensor, cutoff, device)
    else:  # Color image
        filtered_channels = []
        for channel in range(img_tensor.shape[2]):
            filtered_channel = _apply_ideal_low_pass(img_tensor[:, :, channel], cutoff, device)
            filtered_channels.append(filtered_channel)
        filtered_tensor = torch.stack(filtered_channels, dim=2)

    # Move the tensor back to CPU before converting to a PIL image
    filtered_tensor = filtered_tensor.cpu()
    filtered_image = PIL.Image.fromarray(filtered_tensor.numpy().astype(np.uint8))

    del img_tensor, filtered_tensor

    # Free up GPU memory if the device is CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return filtered_image


def _apply_ideal_low_pass(
        channel: torch.Tensor,
        cutoff: int,
        device: torch.device
) -> torch.Tensor:
    """
    Apply a low-pass filter to a single channel of an image.

    Args:
        channel (torch.Tensor): A single channel of the image.
        cutoff (int): The cutoff frequency for the low-pass filter.
        device (torch.device): The device to run the computation on.

    Returns:
        torch.Tensor: The filtered channel after applying the low-pass filter.
    """
    # Perform a 2D Fourier transform
    f = fft.fft2(channel)
    fshift = fft.fftshift(f)

    # Create a low-pass filter
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
    distance = torch.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)
    mask[distance <= cutoff] = 1

    # Apply the filter
    fshift = fshift * mask

    # Perform an inverse Fourier transform
    f_ishift = fft.ifftshift(fshift)
    img_back = fft.ifft2(f_ishift)
    img_back = torch.abs(img_back)

    # Normalize to the range of 0 - 255
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255

    del fshift, f_ishift, distance, mask

    return img_back


def gaussian_high_pass_filter(
        image: PIL.Image.Image,
        cutoff: float,
        device: torch.device = torch.device('cpu')
):
    """
    Apply a Gaussian high-pass filter to a PIL image on the specified device.

    Args:
        image (PIL.Image.Image): Input PIL image.
        cutoff (float): Cutoff frequency for the Gaussian high-pass filter.
        device (torch.device): Device (CPU or GPU) to perform the computation on.

    Returns:
        PIL.Image.Image: Filtered PIL image.
    """
    if cutoff > 1.0:
        cutoff = int(cutoff)
    else:
        cutoff = int(math.sqrt((image.size[0] / 2) ** 2 + (image.size[1] / 2) ** 2) * cutoff)
    if len(image.size) > 2:
        image = image.convert("RGB")
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Get the number of channels in the image
    num_channels = image_np.shape[-1] if len(image_np.shape) == 3 else 1

    filtered_channels = []
    for i in range(num_channels):
        # Extract the current channel
        if num_channels > 1:
            channel = image_np[:, :, i]
        else:
            channel = image_np

        # Convert the channel to a PyTorch tensor and move it to the specified device
        channel_tensor = torch.from_numpy(channel).float().to(device)

        # Get the height and width of the channel
        rows, cols = channel_tensor.shape

        # Create grid coordinates
        u = torch.arange(0, rows).to(device)
        v = torch.arange(0, cols).to(device)
        U, V = torch.meshgrid(u, v, indexing='ij')

        # Calculate the center coordinates
        crow, ccol = rows // 2, cols // 2

        # Calculate the distance in the frequency domain
        D = torch.sqrt((U - crow) ** 2 + (V - ccol) ** 2)

        # Create the Gaussian high-pass filter
        H = 1 - torch.exp(-(D ** 2) / (2 * (cutoff ** 2)))

        # Perform the Fourier transform on the channel
        f = torch.fft.fft2(channel_tensor)
        fshift = torch.fft.fftshift(f)

        # Apply the filter to the frequency domain channel
        fshift = fshift * H

        # Perform the inverse Fourier transform
        f_ishift = torch.fft.ifftshift(fshift)
        channel_back = torch.fft.ifft2(f_ishift)
        channel_back = torch.abs(channel_back)

        # Move the result back to the CPU and convert it to a NumPy array
        channel_back_np = channel_back.cpu().numpy()
        filtered_channels.append(channel_back_np)

    # Combine the filtered channels
    if num_channels > 1:
        filtered_image_np = np.stack(filtered_channels, axis=-1)
    else:
        filtered_image_np = filtered_channels[0]

    # Ensure pixel values are in the range of 0 - 255
    filtered_image_np = np.clip(filtered_image_np, 0, 255).astype(np.uint8)

    # Convert the NumPy array back to a PIL image
    filtered_image_pil = PIL.Image.fromarray(filtered_image_np)

    return filtered_image_pil


def gaussian_low_pass_filter(
        image: PIL.Image.Image,
        cutoff: float,
        device: torch.device = torch.device('cpu')
):
    """
    Apply a Gaussian low-pass filter to a PIL image on the specified device.

    Args:
        image (PIL.Image.Image): Input PIL image.
        cutoff (float): Cutoff frequency for the Gaussian low-pass filter.
        device (torch.device): Device (CPU or GPU) to perform the computation on.

    Returns:
        PIL.Image.Image: Filtered PIL image.
    """
    if cutoff > 1.0:
        cutoff = int(cutoff)
    else:
        cutoff = int(math.sqrt((image.size[0] / 2) ** 2 + (image.size[1] / 2) ** 2) * cutoff)
    if len(image.size) > 2:
        image = image.convert("RGB")
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Get the number of channels in the image
    num_channels = image_np.shape[-1] if len(image_np.shape) == 3 else 1

    filtered_channels = []
    for i in range(num_channels):
        # Extract the current channel
        if num_channels > 1:
            channel = image_np[:, :, i]
        else:
            channel = image_np

        # Convert the channel to a PyTorch tensor and move it to the specified device
        channel_tensor = torch.from_numpy(channel).float().to(device)

        # Get the height and width of the channel
        rows, cols = channel_tensor.shape

        # Create grid coordinates
        u = torch.arange(0, rows).to(device)
        v = torch.arange(0, cols).to(device)
        U, V = torch.meshgrid(u, v, indexing='ij')

        # Calculate the center coordinates
        crow, ccol = rows // 2, cols // 2

        # Calculate the distance in the frequency domain
        D = torch.sqrt((U - crow) ** 2 + (V - ccol) ** 2)

        # Create the Gaussian low-pass filter
        H = torch.exp(-(D ** 2) / (2 * (cutoff ** 2)))

        # Perform the Fourier transform on the channel
        f = torch.fft.fft2(channel_tensor)
        fshift = torch.fft.fftshift(f)

        # Apply the filter to the frequency domain channel
        fshift = fshift * H

        # Perform the inverse Fourier transform
        f_ishift = torch.fft.ifftshift(fshift)
        channel_back = torch.fft.ifft2(f_ishift)
        channel_back = torch.abs(channel_back)

        # Move the result back to the CPU and convert it to a NumPy array
        channel_back_np = channel_back.cpu().numpy()
        filtered_channels.append(channel_back_np)

    # Combine the filtered channels
    if num_channels > 1:
        filtered_image_np = np.stack(filtered_channels, axis=-1)
    else:
        filtered_image_np = filtered_channels[0]

    # Ensure pixel values are in the range of 0 - 255
    filtered_image_np = np.clip(filtered_image_np, 0, 255).astype(np.uint8)

    # Convert the NumPy array back to a PIL image
    filtered_image_pil = PIL.Image.fromarray(filtered_image_np)

    return filtered_image_pil
