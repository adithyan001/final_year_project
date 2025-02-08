import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

n = int(input("Enter the percentage of data (0-100) to introduce errors: ")) #inputs the error percentage

#function to convert rgb/rgba matrix into binary matrix
def rgb_to_binary_matrix(image):
    height, width, channels = image.shape
    binary_matrix = np.empty((height, width), dtype=object)
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            if channels == 3:  # 24-bit RGB
                r, g, b = pixel
                binary_str = f"{r:08b}{g:08b}{b:08b}"
            elif channels == 4:  # 32-bit RGBA
                r, g, b, alpha = pixel
                binary_str = f"{r:08b}{g:08b}{b:08b}{alpha:08b}"
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")

            binary_matrix[i, j] = binary_str
    return binary_matrix,channels

#function to modify the binary matrix for vadf compression
def transform_32bit(binary_str):
    if len(binary_str) == 24:
        part1, part2, part3 = binary_str[:8], binary_str[8:16], binary_str[16:] #separates R,G,B channels
        new_binary = "".join(part1[i] + part2[i] + part3[i] for i in range(8)) #transforming the matrix
        transformed_binary = new_binary + '00000000' #appending 0s to obtain 32-bit pixel
        return transformed_binary

    elif len(binary_str) == 32:
        part1, part2, part3, part4 = binary_str[:8], binary_str[8:16], binary_str[16:24], binary_str[24:]#separates R,G,B, alpha channels
        new_binary = "".join(part1[i] + part2[i] + part3[i] + part4[i] for i in range(8))
        return new_binary

    else:
        raise ValueError("Input binary string must be 24 or 32 bits long.")

#function for reverse transforming to rgb/rgba
def reverse_transform_24bit(transformed_binary,channels):
    if(channels)  == 3:
       original_24bit = transformed_binary[:24]
       part1 = [''] * 8
       part2 = [''] * 8
       part3 = [''] * 8
       for i in range(8):
           part1[i] = original_24bit[i * 3]
           part2[i] = original_24bit[i * 3 + 1]
           part3[i] = original_24bit[i * 3 + 2]

       original_24_binary = ''.join(part1) + ''.join(part2) + ''.join(part3)
       return original_24_binary #24-bit RGB

    elif(channels) == 4:
       original_32bit = transformed_binary[:32]
       part1 = [''] * 8
       part2 = [''] * 8
       part3 = [''] * 8
       part4 = [''] * 8
       for i in range(8):
           part1[i] = original_32bit[i * 4]
           part2[i] = original_32bit[i * 4 + 1]
           part3[i] = original_32bit[i * 4 + 2]
           part4[i] = original_32bit[i * 4 + 3]

       original_32_binary = ''.join(part1) + ''.join(part2) + ''.join(part3) + ''.join(part4)
    #  log_output("Reversed 24-bit: " + original_binary + "\n")
       return original_32_binary #32-bit RGBA

    else:
        raise ValueError("Input binary string must be 24 or 32 bits long.")

def d2b(n, r):
    binary = bin(n)[2:].zfill(r) #decimal to r-bit binary
    return binary


def b2d(binary_string):
    decimal = int(binary_string, 2) #binary to decimal
    return decimal

#vadf encoder
def compress_11bit(num):
  if(num <=1):
    return("00000000000")
  else:
    bnum = d2b(num,32)
    for i in range(len(bnum)):
        if bnum[i] == '1':
            loc = 31 - i     #finding the position of location bit
            break
    bloc = d2b(loc,5)
    #finding the data bits
    if (loc <6):
        data_bits = bnum[-loc:]
        while len(data_bits) < 6:
            data_bits = '0' + data_bits
    elif (loc ==6):
        data_bits = bnum[32 - loc: 38 - loc]
    else:
        data_bits = bnum[32 - loc: 37 - loc]
        if(bnum[38-loc]=='1'):
           data_bits = data_bits + '1'
        else:
           data_bits = data_bits + bnum[37-loc]
    out11 = bloc+ data_bits
    return out11


#function for introducing error
def flip_one_bit(binary_str,n):
    if random.randint(1, 100) > n:
        return binary_str

    binary_list = list(binary_str)
    flip_index = random.randint(0, len(binary_list) - 1)
    binary_list[flip_index] = '0' if binary_list[flip_index] == '1' else '1' #single bit flip for n% pixels
    flipped_binary = ''.join(binary_list)
    return flipped_binary

#vadf decoder
def decompress_11bit(instream):
    if(instream == 0 ):
        outstream = "00000000000000000000000000000000"

    else:
        outstream = ""
        loc = 31 - b2d(instream[0:5])
        for i in range(loc):
           outstream = outstream + '0'
        outstream = outstream + '1'

        if ((31 - loc) < 6 ):
          k = 31 - loc
        else:
          k = 6
        for i in range(k):
            outstream = outstream + instream[i-k]
        while len(outstream) < 32:
            outstream = outstream + '0'
    return b2d(outstream) #32-bit vadf decoded output

#function for image processing
def process_image(image_path, image):
    binary_matrix,channels = rgb_to_binary_matrix(image) #rgb/rgba to binary matrix
    height, width = binary_matrix.shape
    compressed_matrix = np.empty((height, width), dtype=object)
    flipped_matrix = np.empty((height, width), dtype=object)
    decompressed_matrix = np.empty((height, width), dtype=object)
    decompressed_error_matrix = np.empty((height, width), dtype=object)
    reconstructed_matrix = np.empty((height, width), dtype=object)
    reconstructed_error_matrix = np.empty((height, width), dtype=object)

    for i in range(height):
        for j in range(width):
            binary_32bit = transform_32bit(binary_matrix[i, j]) #transformed matrix
            compressed_matrix[i, j] = compress_11bit(int(binary_32bit, 2)) #vadf compressed matrix
            flipped_matrix[i, j] = flip_one_bit(compressed_matrix[i, j],n) #single bit flipped matrix
            decompressed_matrix[i, j] = d2b(decompress_11bit(compressed_matrix[i, j]), 32) #decompressed matrix(without error)
            decompressed_error_matrix[i, j] = d2b(decompress_11bit(flipped_matrix[i, j]), 32) #decompressed matrix(with error)
            reconstructed_matrix[i, j] = reverse_transform_24bit(decompressed_matrix[i, j],channels) #reconstructed_matrix(without error)
            reconstructed_error_matrix[i, j] = reverse_transform_24bit(decompressed_error_matrix[i, j],channels) #reconstructed_matrix(with error)

    if(channels)==3:
       reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
       for i in range(height):
          for j in range(width):
              r = b2d(reconstructed_matrix[i, j][:8])
              g = b2d(reconstructed_matrix[i, j][8:16])
              b = b2d(reconstructed_matrix[i, j][16:])
              reconstructed_image[i, j] = [r, g, b] #reconstructed RGB image(without error)

       reconstructed_error_image = np.zeros((height, width, 3), dtype=np.uint8)
       for i in range(height):
          for j in range(width):
              r = b2d(reconstructed_error_matrix[i, j][:8])
              g = b2d(reconstructed_error_matrix[i, j][8:16])
              b = b2d(reconstructed_error_matrix[i, j][16:])
              reconstructed_error_image[i, j] = [r, g, b] #reconstructed RGB image(with error)

    elif(channels)==4:
        reconstructed_image = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                r = b2d(reconstructed_matrix[i, j][:8])
                g = b2d(reconstructed_matrix[i, j][8:16])
                b = b2d(reconstructed_matrix[i, j][16:24])
                alpha = b2d(reconstructed_matrix[i, j][24:])
                reconstructed_image[i, j] = [r, g, b, alpha] #reconstructed RGBA image(without error)

        reconstructed_error_image = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                r = b2d(reconstructed_error_matrix[i, j][:8])
                g = b2d(reconstructed_error_matrix[i, j][8:16])
                b = b2d(reconstructed_error_matrix[i, j][16:24])
                alpha = b2d(reconstructed_error_matrix[i, j][24:])
                reconstructed_error_image[i, j] = [r, g, b, alpha] #reconstructed RGBA image(with error)

    else:
        raise ValueError("Input binary string must be 24 or 32 bits long.")

    # Display images in RGB/RGBA format
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_image)
    axes[1].set_title("Reconstructed Image VADF compression")
    axes[1].axis("off")

    axes[2].imshow(reconstructed_error_image)
    axes[2].set_title("Reconstructed Error Image VADF compression")
    axes[2].axis("off")

    processed_dir = r"images/outputImages/11Bit"

    image_filename = os.path.basename(image_path)  # Get only the filename
    image_name, image_ext = os.path.splitext(image_filename)

    # Save images in the processed directory
    if(channels)==3: #save images in RGB format
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_reconstructed{image_ext}"),
                cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR))
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_error{image_ext}"),
                cv2.cvtColor(reconstructed_error_image, cv2.COLOR_RGB2BGR))

    elif(channels)==4: #save images in RGBA format
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_reconstructed{image_ext}"), reconstructed_image)
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_error{image_ext}"), reconstructed_error_image)

    plt.show()
    return reconstructed_image,reconstructed_error_image

#function for finding the mean absolute difference
def mean_abs_diff(input_image, output_image_name, output_image):
    diff_image = []
    height, width, channels = input_image.shape
    for i in range(height):
        for j in range(width):
            diff_image.append(abs(input_image[i, j] - output_image[i, j]))
    abs_mean = np.mean(diff_image)

    print(f"Original vs {output_image_name} Mean Absolute Difference: {abs_mean}")
    return abs_mean

# read image
image_path = r"images/inputImages/RGB_24bits_palette_sample_image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load image
bits_per_pixel = image.dtype.itemsize * 8 * image.shape[-1]  # Compute total bits per pixel
if bits_per_pixel == 24:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #read as RGB image

reconstructed_image,reconstructed_error_image = process_image(image_path,image)

mean_abs_diff(image, "reconstructed image", reconstructed_image)
mean_abs_diff(image, "reconstructed error image", reconstructed_error_image)
