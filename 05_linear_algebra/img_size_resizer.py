import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Get the path to the image file
img_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'codding.webp')
img = mpimg.imread(img_path)

print(f"Original image shape:{img.shape}") # Display the original image shape
plt.imshow(img) # Display the image

# Reshape the image to a 2D array (height x (width * channels)) so PCA can be applied
img_reshaped = np.reshape(img, (img.shape[0], -1)) 
print(f"Reshaped image shape:{img_reshaped.shape}")

# Compress the image using PCA by reducing the number of features
pca = PCA(32).fit(img_reshaped)
img_transformed = pca.transform(img_reshaped)
print(f"Transformed image shape:{img_transformed.shape}") # Display the transformed image shape
print(f"Explained variance ratio:{np.sum(pca.explained_variance_ratio_):.4f}") # Display the explained variance ratio