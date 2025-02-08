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
def vadf_enc(num):
    if (num <= 1):
        return ("0000000000000000")
    else:
        bnum = d2b(num, 32)
        for i in range(len(bnum)):
            if bnum[i] == '1':
                loc = 31 - i #finding the position of location bit
                break
        bloc = d2b(loc, 5)
        #finding the data bits
        if (loc < 6):
            data_bits = bnum[-loc:]
            while len(data_bits) < 6:
                data_bits = '0' + data_bits
        elif (loc == 6):
            data_bits = bnum[32 - loc: 38 - loc]
        else:
            data_bits = bnum[32 - loc: 37 - loc]
            if (bnum[38 - loc] == '1'):
                data_bits = data_bits + '1'
            else:
                data_bits = data_bits + bnum[37 - loc]
        #finding the parity bit
        count_ones = data_bits.count('1')
        if (count_ones % 2 == 0):
            parity_bit = '0'
        else:
            parity_bit = '1'
        #finding the error correction bits
        int_loc = [int(bit) for bit in bloc[::-1]]
        error_correction_bits = [0] * 4
        error_correction_bits[3] = int_loc[4] ^ int_loc[2] ^ int_loc[1]
        error_correction_bits[2] = int_loc[4] ^ int_loc[3] ^ int_loc[1]
        error_correction_bits[1] = int_loc[4] ^ int_loc[3] ^ int_loc[2]
        error_correction_bits[0] = int_loc[0]

        temp = ''.join(map(str, error_correction_bits))
        ecb = temp[::-1]

        out16 = parity_bit + bloc + ecb + data_bits #16 bit vadf encoded output
        return out16

#function for introducing error
def flip_one_bit(binary_str,n):
    if random.randint(1, 100) > n:
        return binary_str

    binary_list = list(binary_str)
    flip_index = random.randint(0, len(binary_list) - 1)
    binary_list[flip_index] = '0' if binary_list[flip_index] == '1' else '1' #single bit flip for n% pixels
    flipped_binary = ''.join(binary_list)
    return flipped_binary

#hamming code for location and error correction bits
def hamming_code_decode(instream):
  check = [int(bit) for bit in instream[0:10]]
  #syndrome calculation
  s = [0] * 4
  s[0] = check[6] ^ check[1] ^ check[3] ^ check[4]
  s[1] = check[7] ^ check[1] ^ check[2] ^ check[4]
  s[2] = check[8] ^ check[1] ^ check[2] ^ check[3]
  s[3] = check[9] ^ check[5]
  syn = ''.join(map(str, s))
  inp = [int(bit) for bit in instream]
  if (syn[3] == "1"):
    inp[5] = 1 ^ inp[5]
  if (syn[:-1] == "001"):
    inp[8] = 1 ^ inp[8]
  if (syn[:-1] == "010"):
    inp[7] = 1 ^ inp[7]
  if (syn[:-1] == "100"):
    inp[6] = 1 ^ inp[6]
  if (syn[:-1] == "011"):
    inp[2] = 1 ^ inp[2]
  if (syn[:-1] == "101"):
    inp[3] = 1 ^ inp[3]
  if (syn[:-1] == "110"):
    inp[4] = 1 ^ inp[4]
  if (syn[:-1] == "111"):
    inp[1] = 1 ^ inp[1]
  return ''.join(map(str, inp)) #hamming code corrected output

#vadf decoder
def vadf_dec_ideal(instream):
    if(instream == "0000000000000000" ):
      return 0
    outstream = ""
    loc = 31 - b2d(instream[1:6])
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
    corrected_matrix = np.empty((height, width), dtype=object)
    decompressed_matrix = np.empty((height, width), dtype=object)
    decompressed_error_matrix = np.empty((height, width), dtype=object)
    decompressed_corrected_matrix = np.empty((height, width), dtype=object)
    reconstructed_matrix = np.empty((height, width), dtype=object)
    reconstructed_error_matrix = np.empty((height, width), dtype=object)
    reconstructed_corrected_matrix = np.empty((height, width), dtype=object)

    for i in range(height):
        for j in range(width):
            binary_32bit = transform_32bit(binary_matrix[i, j]) #transformed matrix
            compressed_matrix[i, j] = vadf_enc(int(binary_32bit, 2)) #vadf compressed matrix
            flipped_matrix[i, j] = flip_one_bit(compressed_matrix[i, j],n) #single bit flipped matrix
            corrected_matrix[i, j] = hamming_code_decode(flipped_matrix[i, j]) #hamming code corrected matrix
            decompressed_matrix[i, j] = d2b(vadf_dec_ideal(compressed_matrix[i, j]), 32) #decompressed matrix(without error)
            decompressed_error_matrix[i, j] = d2b(vadf_dec_ideal(flipped_matrix[i, j]), 32) #decompressed matrix(with error)
            decompressed_corrected_matrix[i, j] = d2b(vadf_dec_ideal(corrected_matrix[i, j]), 32) #decompressed matrix(with error correction)
            reconstructed_matrix[i, j] = reverse_transform_24bit(decompressed_matrix[i, j],channels) #reconstructed_matrix(without error)
            reconstructed_error_matrix[i, j] = reverse_transform_24bit(decompressed_error_matrix[i, j],channels) #reconstructed_matrix(with error)
            reconstructed_corrected_matrix[i, j] = reverse_transform_24bit(decompressed_corrected_matrix[i, j],channels) #reconstructed_matrix(with error correction)

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

       reconstructed_corrected_image = np.zeros((height, width, 3), dtype=np.uint8)
       for i in range(height):
          for j in range(width):
              r = b2d(reconstructed_corrected_matrix[i, j][:8])
              g = b2d(reconstructed_corrected_matrix[i, j][8:16])
              b = b2d(reconstructed_corrected_matrix[i, j][16:])
              reconstructed_corrected_image[i, j] = [r, g, b] #reconstructed RGB image(with error correction)

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

        reconstructed_corrected_image = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                r = b2d(reconstructed_corrected_matrix[i, j][:8])
                g = b2d(reconstructed_corrected_matrix[i, j][8:16])
                b = b2d(reconstructed_corrected_matrix[i, j][16:24])
                alpha = b2d(reconstructed_corrected_matrix[i, j][24:])
                reconstructed_corrected_image[i, j] = [r, g, b, alpha] #reconstructed RGB image(with error correction)

    else:
        raise ValueError("Input binary string must be 24 or 32 bits long.")

    # Display images in RGB/RGBA format
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes[0,0].imshow(image)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis("off")

    axes[0,1].imshow(reconstructed_image)
    axes[0,1].set_title("Reconstructed Image VADF compression")
    axes[0,1].axis("off")

    axes[1,0].imshow(reconstructed_error_image)
    axes[1,0].set_title("Reconstructed Error Image VADF compression")
    axes[1,0].axis("off")

    axes[1,1].imshow(reconstructed_corrected_image)
    axes[1,1].set_title("Reconstructed Corrected Image VADF compression")
    axes[1,1].axis("off")

    processed_dir = r"images/outputImages"

    image_filename = os.path.basename(image_path)  # Get only the filename
    image_name, image_ext = os.path.splitext(image_filename)

    # Save images in the processed directory
    if(channels)==3: #save images in RGB format
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_reconstructed{image_ext}"),
                cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR))
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_error{image_ext}"),
                cv2.cvtColor(reconstructed_error_image, cv2.COLOR_RGB2BGR))
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_corrected{image_ext}"),
                cv2.cvtColor(reconstructed_corrected_image, cv2.COLOR_RGB2BGR))

    elif(channels)==4: #save images in RGBA format
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_reconstructed{image_ext}"), reconstructed_image)
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_error{image_ext}"), reconstructed_error_image)
      cv2.imwrite(os.path.join(processed_dir, f"{image_name}_corrected{image_ext}"), reconstructed_corrected_image)

    plt.show()
    return reconstructed_image,reconstructed_error_image,reconstructed_corrected_image

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
image_path = r"images/inputImages/pycharm_icon_32bit.png"


# image_path = os.path.abspath("images/pycharm_icon_32bit.png")
# print(os.path.exists(image_path))
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load image
bits_per_pixel = image.dtype.itemsize * 8 * image.shape[-1]  # Compute total bits per pixel
if bits_per_pixel == 24:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #read as RGB image
reconstructed_image,reconstructed_error_image,reconstructed_corrected_image = process_image(image_path,image)

mean_abs_diff(image, "reconstructed image", reconstructed_image)
mean_abs_diff(image, "reconstructed error image", reconstructed_error_image)
mean_abs_diff(image,"reconstructed corrected image",reconstructed_corrected_image)