import cv2
from experts.deblur_expert import restore_deblur
import numpy as np

res = restore_deblur('datasets/benchmark/blur_test/degraded/bird.png')
print("Deblur completed. Output:", res)

# Check if there is a white blob
img = cv2.imread(res)
if img is not None:
    print("Mean pixel value:", np.mean(img))
    print("Max pixel value:", np.max(img))
    print("Min pixel value:", np.min(img))
    
    # Check center vs edges
    h, w = img.shape[:2]
    center_roi = img[h//2-10:h//2+10, w//2-10:w//2+10]
    edge_roi = img[0:20, 0:20]
    print("Center mean:", np.mean(center_roi))
    print("Edge mean:", np.mean(edge_roi))
