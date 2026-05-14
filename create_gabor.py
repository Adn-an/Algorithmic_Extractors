# from skimage.filters import gabor
# from PIL import Image
# import numpy as np
# from pathlib import Path

# def get_gabor(image, frequency=0.6, theta=0):
#     """
#     Apply Gabor filter to a grayscale image.

#     Args:
#         image (PIL.Image.Image): Input grayscale image.
#         frequency (float): Frequency of the sinusoidal function.
#         theta (float): Orientation of the normal to the parallel stripes of a Gabor function (in radians).

#     Returns:
#         PIL.Image.Image: Gabor filtered image normalized to [0, 255].
#     """
#     # Convert to numpy array
#     image_np = np.array(image)
#     # Apply Gabor filter
#     filt_real, filt_imag = gabor(image_np, frequency=frequency, theta=theta)
#     # Use magnitude of the response
#     magnitude = np.sqrt(filt_real**2 + filt_imag**2)
#     # Normalize to [0, 1]
#     magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
#     # Convert back to PIL image (uint8)
#     gabor_image = Image.fromarray((magnitude_norm * 255).astype(np.uint8))
#     return gabor_image

# # Frequencies and orientations
# frequencies = [0.9, 0.6, 0.3, 0.1, 0.05]
# thetas = [0, 0.45, 0.9]

# # # Step 1: Create folder structure
# # for f in frequencies:
# #     for t in thetas:
# #         base_folder = f"GABOR_f{f}_t{t}"
# #         for sub in ["train", "test"]:
# #             Path(f"{base_folder}/{sub}").mkdir(parents=True, exist_ok=True)

# # Step 2: Process images
# for folder in ['train', 'test']:
#     image_dir = Path(f"./{folder}")
#     for image_path in image_dir.glob("*.jpg"):
#         image = Image.open(image_path).convert("L")  # Ensure grayscale
#         for f in frequencies:
#             for t in thetas:
#                 gabor_image = get_gabor(image, frequency=f, theta=t)
#                 out_folder = f"GABOR_f{f}_t{t}/{folder}"
#                 out_path = Path(out_folder) / image_path.name
#                 gabor_image.save(out_path)


import cupy as cp
from cupyx.scipy.ndimage import convolve
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Frequencies and orientations
frequencies = [0.9, 0.6, 0.3, 0.1, 0.05]
thetas = [0, 0.45, 0.9]

def create_gabor_kernel(frequency, theta, sigma_x=3.0, sigma_y=3.0, size=31):
    """Create a Gabor kernel on the GPU using CuPy."""
    half_size = size // 2
    x, y = cp.meshgrid(cp.arange(-half_size, half_size + 1), cp.arange(-half_size, half_size + 1))
    
    # Rotate coordinates
    x_theta = x * cp.cos(theta) + y * cp.sin(theta)
    y_theta = -x * cp.sin(theta) + y * cp.cos(theta)
    
    gb = cp.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) \
         * cp.cos(2 * cp.pi * frequency * x_theta)
    return gb

def apply_gabor_gpu(image_np, frequency, theta):
    """Apply Gabor filter on GPU using CuPy."""
    # Convert to CuPy
    img_cp = cp.asarray(image_np, dtype=cp.float32) / 255.0

    # Create Gabor kernel
    kernel = create_gabor_kernel(frequency, theta)

    # Convolve
    filtered = convolve(img_cp, kernel, mode='reflect')
    
    # Normalize and convert back to uint8
    filtered = cp.abs(filtered)
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-8) * 255
    return cp.asnumpy(filtered).astype(np.uint8)

def process_image(image_path, folder):
    """Load image, apply Gabor filters on GPU, and save results."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    for f in frequencies:
        for t in thetas:
            try:
                filtered = apply_gabor_gpu(image, f, t)
                out_dir = Path(f"GABOR_f{f}_t{t}/{folder}")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / image_path.name
                cv2.imwrite(str(out_path), filtered)
            except Exception as e:
                print(f"Failed processing {image_path} with f={f}, t={t}: {e}")

# Main loop
for folder in ['train', 'test']:
    image_dir = Path(folder)
    images = list(image_dir.glob("*.jpg"))
    
    for image_path in tqdm(images, desc=f"Processing {folder} images"):
        process_image(image_path, folder)