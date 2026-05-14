import os
import matplotlib.pyplot as plt
from skimage import io, feature, color, exposure, util

# --- Configuration ---
# 1. Folder to read images from
INPUT_FOLDER = "./train"

# 2. Folder to save HOG visualizations to
OUTPUT_FOLDER = "./HOG/train"
# ---

def batch_compute_hog(input_dir, output_dir):
    """
    Reads all images from input_dir, computes their HOG visualization,
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
    print(f"Saving HOG images to: {output_dir}\n")

    # 3. Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # 4. Check if the file is a common image type
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            print(f"Skipping {filename} (not a recognized image file).")
            continue

        # 5. Construct full file paths
        img_path = os.path.join(input_dir, filename)

        try:
            # 6. Read the image. 
            # as_gray=True converts color images to grayscale automatically.
            # If the image is already grayscale, it will be loaded correctly.
            image = io.imread(img_path, as_gray=True)

            print(f"Processing {filename}...")

            # 7. Compute HOG features and the visualization image
            # fd is the feature descriptor (a 1D numpy array)
            # hog_image is the 2D visualization
            fd, hog_image = feature.hog(
                image,
                orientations=9,          # Number of orientation bins
                pixels_per_cell=(8, 8),  # Size (in pixels) of a cell
                cells_per_block=(2, 2),  # Number of cells in each block
                visualize=True           # Return the visualization
            )

            # 8. Enhance contrast of the HOG image for better visibility
            # This stretches the intensity values to span 0-255
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range='image')
            
            # 9. Convert to 8-bit unsigned integer (required for saving)
            hog_image_uint8 = util.img_as_ubyte(hog_image_rescaled)

            # 10. Create a new filename for the HOG image
            base_filename, _ = os.path.splitext(filename)
            hog_filename = f"{base_filename}.png"
            output_path = os.path.join(output_dir, hog_filename)

            # 11. Save the HOG visualization
            io.imsave(output_path, hog_image_uint8)
            print(f"Successfully saved {hog_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    batch_compute_hog(INPUT_FOLDER, OUTPUT_FOLDER)
