import numpy as np
import random
import time
import constants as c
import cv2
import os
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F

#README: questo codice prende una rete pretrainata da torchvision (ResNet50), estrae le feature da un'immagine campionata randomicamente e la confronta con un subset..
#..(che deve diventare l'insieme delle immagini della mappa) e ne calcola la distanza euclidea. Usa le notazioni del file constants di Giorgia. Gli ultimi 2 layer della resnet..
#..sono rimossi perché ci interessa la feature map e non le predizioni :)

# Funzione per prendere un'immagine casualmente da una cartella
def random_img():
    image_files = [file for file in os.listdir(c.FOLDER_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("Nessun file immagine trovato nel percorso specificato.")
        return None

    random_image = random.choice(image_files)
    full_image_path = os.path.join(c.FOLDER_PATH, random_image)
    example_image = cv2.imread(full_image_path)

    if example_image is None:
        print(f"Errore nel caricamento dell'immagine: {full_image_path}. Verifica il percorso e l'integrità del file.")
        return None

    # Converti da BGR (OpenCV) a RGB (PIL)
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    # Converti il numpy.ndarray in un oggetto PIL.Image
    example_image = Image.fromarray(example_image)
    return full_image_path
    #return example_image

#subset del dataset per velocizzare l'esecuzione
def get_random_subset(folder_path, num_samples):
    all_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
    if num_samples > len(all_files):
        return all_files  # ritorna tutti i file se il num_samples è più grande del numero di file disponibili
    else:
        return random.sample(all_files, num_samples)


def compare_images(feature1, feature2):
    distance = F.pairwise_distance(feature1, feature2)
    return distance

# Definizione delle trasformazioni per il modello
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Dimensione richiesta da ResNet
    transforms.ToTensor(),  # Converti in tensore PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizzazione consigliata ImageNet
])

def to_tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t

def extract_features(image_tensor):
    with torch.no_grad():
        # Ottenere le output fino all'ultimo strato convoluzionale
        features = model(image_tensor)
        # Applicare pooling globale per ridurre le dimensioni spaziali
        features = features.view(features.size(0), -1)
        return features

def compare_images(features1, features2):
    return torch.norm(features1 - features2, p=2)


#_____________________________________________________________________________________________________

# Carica il modello ResNet preaddestrato
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-2]) # Rimuove l'ultimo fc layer e l'ultimo pooling layer

# Ottieni un'immagine casuale, visualizzala e trasformala
img_path = random_img()
if img_path is not None:
    
    img_tensor = to_tensor(img_path)
    print("ok")
else:
    print("Nessuna immagine da visualizzare.")

#estrazione delle feature
features1=extract_features(img_tensor)

#calcolo immagine più vicina
min_distance = float('inf')
min_file_path = None
subset_files = get_random_subset(c.FOLDER_PATH, 100)
# Itera su tutte le immagini nel dataset
for file in subset_files:
    if file.endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(c.FOLDER_PATH, file)  
        image_to_compare = to_tensor(file_path)
        features2 = extract_features(image_to_compare)

        # Calcola la distanza tra le feature dell'immagine di riferimento e quelle correnti
        distance = compare_images(features1, features2)

        # Controlla se la distanza trovata è la minima e aggiorna se necessario
        if distance < min_distance:
            min_distance = distance
            min_file_path = file_path

# Stampa i risultati
if min_file_path:
    print(f'Image with minimum distance: {min_file_path}, Distance: {min_distance}')

img = cv2.imread(img_path)

# Controlla se l'immagine è stata caricata correttamente
if img is not None:
    # Mostra l'immagine
    cv2.imshow('Immagine', img)
    # Attendi che l'utente prema un tasto per chiudere la finestra
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Errore: l'immagine non è stata trovata o non è possibile leggerla.")

# Carica l'immagine utilizzando OpenCV
img = cv2.imread(min_file_path)

# Controlla se l'immagine è stata caricata correttamente
if img is not None:
    # Mostra l'immagine
    cv2.imshow('Immagine', img)
    # Attendi che l'utente prema un tasto per chiudere la finestra
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Errore: l'immagine non è stata trovata o non è possibile leggerla.")