import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("/Users/ayoub/Documents/GitHub/yolo_comparaison/models/seg_160_images_Colin.pt")

# On fait l'inference sur une image et on stocke les resultats
image_path = "/Users/ayoub/Documents/GitHub/yolo_comparaison/images_Colin/0c19f409-c844-4e2b-9065-e9d843cddbac_98_0890四部丛刊·欧阳文忠公文集五.png"
results = model.predict(source=image_path)

# On load notre image
image = cv2.imread(image_path)
orig_height, orig_width = image.shape[:2]

# On extrait les prédictions du modèle + on vérifie s'il y a des masques et des bounding boxes
predictions = results[0]
print(f"Predictions: {predictions}")

# Check if there are masks in the predictions
if hasattr(predictions, 'masks') and predictions.masks is not None:
    # On convertit les masques en numpy array
    masks = predictions.masks.data.cpu().numpy()
    
    for mask in masks:
        # On resize les masques pour les adapter à la taille de l'image
        mask_resized = cv2.resize(mask, (orig_width, orig_height))
        mask_resized = (mask_resized * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Et on dessinne les contours
        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
