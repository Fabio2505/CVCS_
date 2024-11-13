import numpy as np
import torch
import map as mp
import funzione_mappa as fp
import config_constants as c
import retrieval_CV as ret
import Rete as Net
import os
import cv2
from torchvision import models, transforms
from PIL import Image
import random
from ANSRGBmodel_utils import ANSRGB, depth_img_preprocessing
from object_detection import create_object_mask, find_object_areas, calculate_average_distance


def show_image(image):
    if image is not None:
        # Visualizza l'immagine (opzionale)
        cv2.imshow('Random Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Non è stata trovata o caricata nessuna immagine.")
    return None


def insert_tag(image):  # funzione che valuta se l'erba è tagliata o meno
    image = cv2.Canny(image, 10, 50)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)  # aggiungi channel_in
    output = model(image)
    _, stato_erba = torch.max(output, dim=1)
    stato_erba = stato_erba.item()
    return stato_erba


def insert_occlusion(image):
    image_segm_input = image

    binary_mask, resized_image = create_object_mask(image_segm_input)
    areas = find_object_areas(binary_mask)

    if areas:
        img_model_input = depth_img_preprocessing(image, (3, 128, 128))
        with torch.no_grad():
            depth_output = depth_model({"rgb": img_model_input})

        depth_model_result = depth_output["occ_estimate"]

        depth_image = depth_model_result[:, 0, :, :].squeeze().numpy()
        average_distance = calculate_average_distance(depth_image, areas)
        # print(f"Distanza media: {average_distance:.3f}")
        if average_distance < 0.5:
            return 1
        else:
            return 0
    else:
        # print("non ci sono ostacoli")
        return 0


def controlla_giardino(mappa, image_directory):
    giardino_noto = 0

    # Carica l'immagine campione
    sample = ret.new_image(image_files)  # Assume che `new_image` prenda la mappa e restituisca un'immagine
    Resnet = ret.istanzia_modello()  # Istanzia il modello ResNet

    # Verifica che l'immagine campione sia un array NumPy e che la mappa esista
    if isinstance(sample, np.ndarray) and mappa is not None:
        sample = Image.fromarray(sample)

    # trasformazioni richieste da ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Dimensione richiesta da ResNet
        transforms.ToTensor(),  # Converti in tensore PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizzazione ImageNet
    ])

  
    sample_transformed = transform(sample)

    sample_transformed = sample_transformed.unsqueeze(0)
    print(f"Dimensione del tensore sample: {sample_transformed.size()}")

    # Estrai le features dell'immagine campione usando ResNet
    sample_features = ret.extract_features(sample_transformed, Resnet)

    # Ottieni la lista di tutte le immagini nella directory specificata
    immagini = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

    for immagine_file in immagini:
        # Carica l'immagine corrente
        immagine_path = os.path.join(image_directory, immagine_file)
        immagine = Image.open(immagine_path)

        # Applica le trasformazioni all'immagine corrente
        immagine_transformed = transform(immagine)
        immagine_transformed = immagine_transformed.unsqueeze(0)

        # Estrai le caratteristiche dell'immagine corrente
        immagine_features = ret.extract_features(immagine_transformed, Resnet)

        # Confronta le caratteristiche dell'immagine campione con quelle dell'immagine corrente
        similarity = ret.compare_images(sample_features, immagine_features)

        if similarity > 0.6:
            giardino_noto = 1
            break

    return giardino_noto


# CORE


image_files = [file for file in os.listdir(c.IMAGES_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]
mappa = mp.Map(c.MAP_WIDTH, c.MAP_HEIGHT, c.CELL_SIZE)
model = Net.CNN()
model.load_state_dict(torch.load("cnn.pth"))

# inizializzazione modello ANSRGB e caricamento dei pesi
depth_model = ANSRGB()
state_dict = torch.load('ckpt.12.pth', map_location=torch.device('cpu'))
depth_model.load_state_dict(state_dict, strict=False)
depth_model.eval()

memory_path = c.HOME_PATH
image_directory = os.path.join(memory_path, 'immagini_mappa')
json_file = os.path.join(memory_path, 'image_paths.json')
json_tags_file = os.path.join(memory_path, 'tags_file.json')
json_occlusion_file = os.path.join(memory_path, 'occlusion_file.json')  # FILE PER SALVARE LE OCCLUSIONI (DEPTH)


if (controlla_giardino(mappa, image_directory)):  # Verifica che 'json_file' esista 
    # Carica la mappa dal file
    fp.load_image_paths_from_json(mappa, json_file)
    fp.load_tags_from_json(mappa, json_tags_file)
    fp.load_occlusions_from_json(mappa, json_occlusion_file)
    print('percorso caricato')


else:  

    mappa = mp.Map(c.MAP_WIDTH, c.MAP_HEIGHT, c.CELL_SIZE)
    fp.riempi_mappa(mappa)  # scrive i percorsi nella mappa e salva le foto nella cartella
    fp.trasforma_griglia(mappa)  # per il file json
    fp.save_image_paths_to_json(mappa, json_file)
    fp.save_tags_to_json(mappa, json_tags_file)
    fp.save_occlusions_to_json(mappa,json_occlusion_file)

mappa.display_map()  # stampa i percorsi nella cartella concatenati
mappa.print_tag_grid()
mappa.print_occlusion_grid()

print(f"MAP_WIDTH: {c.MAP_WIDTH}, MAP_HEIGHT: {c.MAP_HEIGHT}")
