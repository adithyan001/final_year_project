import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#delta encoding
def delta_encode(image):

    image_array = np.array(image, dtype=np.uint8)
    R, G, B = cv2.split(image_array) #split R,G,B channels

    #create delta matrices for R,G,B
    delta_R = np.zeros_like(R, dtype=np.int16)
    delta_G = np.zeros_like(G, dtype=np.int16)
    delta_B = np.zeros_like(B, dtype=np.int16)

    #first pixel is unchanged
    delta_R[0, 0] = R[0, 0]
    delta_G[0, 0] = G[0, 0]
    delta_B[0, 0] = B[0, 0]

    #appling delta transformation for rest of the pixels
    for i in range(1, R.shape[0]):
        delta_R[i, 0] = np.int16(R[i, 0]) - np.int16(R[i - 1, 0])
        delta_G[i, 0] = np.int16(G[i, 0]) - np.int16(G[i - 1, 0])
        delta_B[i, 0] = np.int16(B[i, 0]) - np.int16(B[i - 1, 0])

    for i in range(R.shape[0]):
        for j in range(1, R.shape[1]):
            delta_R[i, j] = np.int16(R[i, j]) - np.int16(R[i, j - 1])
            delta_G[i, j] = np.int16(G[i, j]) - np.int16(G[i, j - 1])
            delta_B[i, j] = np.int16(B[i, j]) - np.int16(B[i, j - 1])

    delta_image = np.stack([delta_R, delta_G, delta_B], axis=-1).astype(np.int16)

    return delta_image

#delta decoding
def delta_decode(delta_image):
    delta_R, delta_G, delta_B = cv2.split(delta_image.astype(np.int16))

    R = np.zeros_like(delta_R, dtype=np.int16)
    G = np.zeros_like(delta_G, dtype=np.int16)
    B = np.zeros_like(delta_B, dtype=np.int16)

    #reconstructing the original image from delta image
    R[0, 0] = delta_R[0, 0]
    G[0, 0] = delta_G[0, 0]
    B[0, 0] = delta_B[0, 0]

    for i in range(1, delta_R.shape[0]):
        R[i, 0] = R[i - 1, 0] + delta_R[i, 0]
        G[i, 0] = G[i - 1, 0] + delta_G[i, 0]
        B[i, 0] = B[i - 1, 0] + delta_B[i, 0]

    for i in range(delta_R.shape[0]):
        for j in range(1, delta_R.shape[1]):
            R[i, j] = R[i, j - 1] + delta_R[i, j]
            G[i, j] = G[i, j - 1] + delta_G[i, j]
            B[i, j] = B[i, j - 1] + delta_B[i, j]

    decoded_image = np.stack([R, G, B], axis=-1).astype(np.uint8)

    return decoded_image



# Load image
image_path = r"images/inputImages/RGB_24bits_palette_sample_image.jpg"
original_image = cv2.imread(image_path)

# Apply delta encoding
delta_image = delta_encode(original_image)

# Apply delta decoding
reconstructed_image = delta_decode(delta_image)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(delta_image)
ax[1].set_title("Delta Encoded Image")
ax[1].axis("off")
ax[2].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
ax[2].set_title("Reconstructed Image")
ax[2].axis("off")
plt.show()