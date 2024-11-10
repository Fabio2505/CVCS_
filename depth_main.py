"""
    1. creare un virtual env con conda
    conda create -n habitat python=3.9 cmake=3.14.0
    conda activate habitat
    2. installare opencv (io versione 4.10.0)
    3. installare habitat-sim e habitat-lab con le istruzioni del github
    conda install habitat-sim -c conda-forge -c aihabitat
    git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e habitat-lab
    pip install -e habitat-baselines
    4. installare pytorch (io versione 2.2.2) + altre librerie se necessario
    5. nel file default.py (percorso: habitat-lab/habitat-lab/habitat/config/default.py) aggiungere la classe Config

    class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

"""

from ANSRGBmodel_utils import ANSRGB, depth_img_preprocessing, compute_depth_values
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from object_detection import create_object_mask, find_object_areas, calculate_average_distance

path = 'img/ajesdgalbf.jpg'  # sostituire con percorso immagine

# 1. preprocessing immagine
img_model_input = depth_img_preprocessing(path, (3, 128, 128))  # sostituire con immagine random o percorso immagine
image_segm_input = path

# 2. inizializzazione modello e caricamento dei pesi
model = ANSRGB()
state_dict = torch.load('ckpt.12.pth', map_location=torch.device('cpu'))
# print(state_dict.keys())
model.load_state_dict(state_dict, strict=False)
model.eval()

with torch.no_grad():
    output = model({"rgb": img_model_input})


model_result = output["occ_estimate"]
print(model_result.shape)  # (1, 2, 128, 128)

depth_image = model_result[:, 0, :, :].squeeze().numpy()

binary_mask, resized_image = create_object_mask(image_segm_input)

cv2.imshow('Tree Trunk Mask', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

areas = find_object_areas(binary_mask)

if areas:
    average_distance = calculate_average_distance(depth_image, areas)
    print(f"Distanza media: {average_distance:.3f}")
else:
    print("non ci sono ostacoli")

"""
# 3. visualizzazione risultati profondità con opencv
depth_map_channel_0 = model_result[0, 0, :, :].detach().cpu().numpy()
depth_map_channel_1 = model_result[0, 1, :, :].detach().cpu().numpy()

# Normalizza entrambe le mappe di profondità per la visualizzazione (range 0-255)
depth_map_channel_0_normalized = cv2.normalize(depth_map_channel_0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_map_channel_1_normalized = cv2.normalize(depth_map_channel_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Canale 0', depth_map_channel_0_normalized)
cv2.imshow('Canale 1', depth_map_channel_1_normalized)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 3bis visualizzazione dei risultati profondità con matplotlib

output_channel_0 = model_result[0, 0, :, :].detach().cpu().numpy()
output_channel_1 = model_result[0, 1, :, :].detach().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(output_channel_0, cmap='gray')  # Canale 0
plt.title('Output Channel 0')

plt.subplot(1, 2, 2)
plt.imshow(output_channel_1, cmap='gray')  # Canale 1
plt.title('Output Channel 1')

plt.show()


# 4.  Calcola valore medio della mappa di profondità
depth_stats_0 = compute_depth_values(model_result, 0)

print(f"Mean Depth: {depth_stats_0['mean_depth']:.3f}")
print(f"Max Depth: {depth_stats_0['max_depth']:.3f}")
print(f"Min Depth: {depth_stats_0['min_depth']:.3f}")


depth_map_channel_1 = model_result[:, 1, :, :]
print("Valore minimo:", torch.min(depth_map_channel_1))
print("Valore massimo:", torch.max(depth_map_channel_1))
"""





