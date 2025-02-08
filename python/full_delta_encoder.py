import numpy as np
import cv2
import matplotlib.pyplot as plt

#delta encoder

def delta_encode(image):

    image_array = np.array(image, dtype=np.uint8)
    if(channels == 4):
        R, G, B, alpha = cv2.split(image_array)
        # R G B Alpha channel seperation
        delta_R = np.zeros_like(R, dtype=np.int16)
        delta_G = np.zeros_like(G, dtype=np.int16)
        delta_B = np.zeros_like(B, dtype=np.int16)
        delta_alpha = np.zeros_like(alpha, dtype=np.int16)

        # First pixel remains the same
        delta_R[0, 0] = R[0, 0]
        delta_G[0, 0] = G[0, 0]
        delta_B[0, 0] = B[0, 0]
        delta_alpha[0, 0] = alpha[0, 0]

        # Row-wise delta encoding
        for i in range(1, R.shape[0]):
            delta_R[i, 0] = np.int16(R[i, 0]) - np.int16(R[i - 1, 0])
            delta_G[i, 0] = np.int16(G[i, 0]) - np.int16(G[i - 1, 0])
            delta_B[i, 0] = np.int16(B[i, 0]) - np.int16(B[i - 1, 0])
            delta_alpha[i, 0] = np.int16(alpha[i, 0]) - np.int16(alpha[i - 1, 0])

        # Column-wise delta encoding
        for i in range(R.shape[0]):
            for j in range(1, R.shape[1]):
                delta_R[i, j] = np.int16(R[i, j]) - np.int16(R[i, j - 1])
                delta_G[i, j] = np.int16(G[i, j]) - np.int16(G[i, j - 1])
                delta_B[i, j] = np.int16(B[i, j]) - np.int16(B[i, j - 1])
                delta_alpha[i, j] = np.int16(alpha[i, j]) - np.int16(alpha[i, j - 1])

        delta_image = np.stack([delta_R, delta_G, delta_B, delta_alpha], axis=-1).astype(np.int16)

    else:
        R, G, B = cv2.split(image_array)
        # R G B  channel seperation

        delta_R = np.zeros_like(R, dtype=np.int16)
        delta_G = np.zeros_like(G, dtype=np.int16)
        delta_B = np.zeros_like(B, dtype=np.int16)

        delta_R[0, 0] = R[0, 0]
        delta_G[0, 0] = G[0, 0]
        delta_B[0, 0] = B[0, 0]

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


def delta_decode(delta_image):

    if (channels == 4):

        delta_R, delta_G, delta_B, delta_alpha = cv2.split(delta_image.astype(np.int16))
        #  delta_R, delta_G, delta_B, delta_alpha seperation

        R = np.zeros_like(delta_R, dtype=np.int16)
        G = np.zeros_like(delta_G, dtype=np.int16)
        B = np.zeros_like(delta_B, dtype=np.int16)
        alpha = np.zeros_like(delta_alpha, dtype=np.int16)

        R[0, 0] = delta_R[0, 0]
        G[0, 0] = delta_G[0, 0]
        B[0, 0] = delta_B[0, 0]
        alpha[0, 0] = delta_alpha[0, 0]

        for i in range(1, delta_R.shape[0]):
            R[i, 0] = R[i - 1, 0] + delta_R[i, 0]
            G[i, 0] = G[i - 1, 0] + delta_G[i, 0]
            B[i, 0] = B[i - 1, 0] + delta_B[i, 0]
            alpha[i,0] = alpha[i - 1, 0] + delta_alpha[i, 0]

        for i in range(delta_R.shape[0]):
            for j in range(1, delta_R.shape[1]):
                R[i, j] = R[i, j - 1] + delta_R[i, j]
                G[i, j] = G[i, j - 1] + delta_G[i, j]
                B[i, j] = B[i, j - 1] + delta_B[i, j]
                alpha[i, j] = alpha[i, j - 1] + delta_alpha[i, j]
        decoded_image = np.stack([R, G, B, alpha], axis=-1).astype(np.uint8)
    else:
        delta_R, delta_G, delta_B = cv2.split(delta_image.astype(np.int16))
        #  delta_R, delta_G, delta_B seperation

        R = np.zeros_like(delta_R, dtype=np.int16)
        G = np.zeros_like(delta_G, dtype=np.int16)
        B = np.zeros_like(delta_B, dtype=np.int16)

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




image_path = r"images/inputImages/RGB_24bits_palette_sample_image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
height,width,channels = image.shape

delta_image = delta_encode(image)

delta_image = delta_encode(image)

reconstructed_image = delta_decode(delta_image)


cv2.imwrite("images/outputImages/delta/delta_encoded.png", delta_image.astype(np.int16))

# writes images as .png if 32bit and .jpg if 24bit RGB

if channels == 4:
    cv2.imwrite("images/outputImages/delta/delta_reconstructed_image.png", reconstructed_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
else:
    cv2.imwrite("images/outputImages/delta/delta_reconstructed_image.jpg", reconstructed_image, [cv2.IMWRITE_JPEG_QUALITY, 100])


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(delta_image)
ax[1].set_title("Delta Encoded Image")
ax[1].axis("off")
ax[2].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
ax[2].set_title("Reconstructed Image")
ax[2].axis("off")
plt.show()