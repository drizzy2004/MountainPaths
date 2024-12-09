from PIL import Image
import numpy as np

def tiff_to_dat(tiff_file, dat_file):
    # Step 1: Open the TIFF file
    tiff_image = Image.open(tiff_file)

    # Step 2: Convert to grayscale if not already
    grayscale_image = tiff_image.convert("L")

    # Step 3: Convert image to a NumPy array
    pixel_data = np.array(grayscale_image)

    # Step 4: Write to .dat file
    with open(dat_file, "w") as file:
        for row in pixel_data:
            file.write(" ".join(map(str, row)) + "\n")

    print(f"Converted {tiff_file} to {dat_file} successfully!")

# Example usage
tiff_to_dat("exportImage.tiff", "kilimanjaro.dat")
