import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import skfuzzy as fuzz

image_path = 'bone enhancement and segmentation/images (1).jpg'  
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

#grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Filtering
filtered_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
median_filtered_image = cv2.medianBlur(gray_image, 5)

#Enhanced Image
enhanced_image = exposure.equalize_adapthist(
    median_filtered_image, clip_limit=0.03)  
plt.figure()
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')
plt.show()

# segmentation
image_flat = enhanced_image.reshape((-1, 1))
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    image_flat.T, c=3, m=2, error=0.005, maxiter=1000, init=None
)
cluster_map = u.argmax(axis=0).reshape(gray_image.shape)

plt.figure()
plt.imshow(cluster_map, cmap='viridis')
plt.title('Segmented Image')
plt.axis('off')
plt.show()
