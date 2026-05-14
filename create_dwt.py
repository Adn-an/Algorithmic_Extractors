import os
import numpy as np
import pywt
from skimage import io, util, exposure

# --- Configuration ---
# 1. Folder to read images from
# INPUT_FOLDER = "input_images"

# 2. Folder to save DWT visualizations to
# OUTPUT_FOLDER = "output_dwt_images"

# 3. Wavelet to use (e.g., 'haar', 'db1', 'db4', 'sym4')
# WAVELET_TYPE = 'haar'
WAVELET_LIST = [
    # 1. Haar: The simplest, very blocky
    'haar',        
    
    # 2. Daubechies: Orthogonal, increasing smoothness
    'db2',
    'db4',         # A common choice
    'db8',         # Smoother
    
    # 3. Symlets: More symmetrical Daubechies
    'sym4',
    'sym8',
    
    # 4. Coiflets: Good for compression
    'coif1',
    'coif5',
    
    # 5. Biorthogonal: Symmetric, good for images (used in JPEG2000)
    'bior1.3',     # Simple, near-Haar
    'bior2.2',     # Simple, linear phase
    'bior4.4',     # A very common choice
    
    # 6. Reverse Biorthogonal
    'rbio2.2',
    
    # 7. Discrete Meyer
    'dmey'
]

# ---

def scale_to_ubyte(array):
    """
    Scales a NumPy array's values to the 0-255 range (uint8)
    for image visualization.
    """
    # Rescale to 0.0 - 1.0
    scaled_array = exposure.rescale_intensity(array, in_range='image')
    # Convert to 0 - 255
    return util.img_as_ubyte(scaled_array)

def batch_compute_dwt(input_dir, output_dir, wavelet):
    """
    Reads all images from input_dir, computes their DWT visualization,
    and saves them to output_dir.
    """
    
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input folder '{input_dir}' not found.")
        print("Please create it and add your images.")
        return

    print(f"Reading images from: {input_dir}")
    print(f"Saving DWT images to: {output_dir}\n")

    # 3. Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # 4. Check if the file is a common image type
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            print(f"Skipping {filename} (not a recognized image file).")
            continue

        # 5. Construct full file paths
        img_path = os.path.join(input_dir, filename)

        try:
            # 6. Read the image as grayscale (DWT works on 2D arrays)
            image = io.imread(img_path, as_gray=True)
            
            # 7. Convert to float (better precision for transforms)
            image = util.img_as_float(image)
            
            # print(f"Processing {filename}...")

            # 8. Apply a single-level 2D DWT
            # This returns:
            # cA = Approximation coefficients (LL)
            # cH = Horizontal detail (LH)
            # cV = Vertical detail (HL)
            # cD = Diagonal detail (HH)
            coeffs = pywt.dwt2(image, wavelet)
            cA, (cH, cV, cD) = coeffs
            
            # Combine the magnitudes (absolute values)
            feature_map = np.abs(cH) + np.abs(cV) + np.abs(cD)
            
            # Scale the combined map to 0-255 for saving
            vis_image = scale_to_ubyte(feature_map)
            
            # 10. Create a new filename for the DWT image
            base_filename, _ = os.path.splitext(filename)
            # (I noticed you removed the _dwt_... part, I kept that change)
            dwt_filename = f"{base_filename}.png"
            output_path = os.path.join(output_dir, dwt_filename)


            # ---------- TILING LOGIC ------------------------
            # 9. Create the visualization image by tiling the 4 components
            
            # Get the shape of the approximation (it will be ~half the original)
            # h_c, w_c = cA.shape

            # Create an empty image to hold the tiled components
            # We make it twice the size of the coefficient maps
            # vis_image = np.zeros((h_c * 2, w_c * 2), dtype=np.uint8)

            # --- Tile and scale each component independently ---
            # Top-left: Approximation (LL) - Scaled
            # vis_image[0:h_c, 0:w_c] = scale_to_ubyte(cA)
            
            # Top-right: Horizontal Detail (LH) - Scaled
            # vis_image[0:h_c, w_c : w_c * 2] = scale_to_ubyte(cH)
            
            # Bottom-left: Vertical Detail (HL) - Scaled
            # vis_image[h_c : h_c * 2, 0:w_c] = scale_to_ubyte(cV)
            
            # Bottom-right: Diagonal Detail (HH) - Scaled
            # vis_image[h_c : h_c * 2, w_c : w_c * 2] = scale_to_ubyte(cD)


            # 10. Create a new filename for the DWT image
            # base_filename, _ = os.path.splitext(filename)
            # dwt_filename = f"{base_filename}.png"
            # output_path = os.path.join(output_dir, dwt_filename)
            # ---------- END TILING LOGIC ------------------------

            # 11. Save the DWT visualization
            io.imsave(output_path, vis_image)
            # print(f"Successfully saved {dwt_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    for WAVELET_TYPE in WAVELET_LIST:
        print(f'processing wavelet type: {WAVELET_TYPE}') 
        for INPUT_FOLDER in ['train_segmented', 'test_segmented']:
            OUTPUT_FOLDER = f'./DWT_{WAVELET_TYPE}/{INPUT_FOLDER}'
            batch_compute_dwt(INPUT_FOLDER, OUTPUT_FOLDER, WAVELET_TYPE)
