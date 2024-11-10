import os
import numpy as np
import config_constants as c
import map as mp
import random
import cv2
import torch
import json
import Rete as Net
from ANSRGBmodel_utils import ANSRGB, depth_img_preprocessing
from object_detection import create_object_mask, find_object_areas, calculate_average_distance

image_files = [file for file in os.listdir(c.IMAGES_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]
memory_path = c.HOME_PATH
image_directory = os.path.join(memory_path, 'immagini_mappa')


def new_image(image_files):
    if not image_files:
        print("Nessun file immagine trovato nel percorso specificato.")
        return None

    random_image = random.choice(image_files)
    full_image_path = os.path.join(c.IMAGES_PATH, random_image)
    example_image = cv2.imread(full_image_path)
    return example_image


def riempi_mappa(mappa):  # base per eventuale navigazione
    for y in range(c.MAP_HEIGHT):
        # condizione ingrandimento
        for x in range(c.MAP_WIDTH):
            # condizione ingrandimento
            image = new_image(image_files)
            img_occlusion = image
            image_to_store = cv2.resize(image, (224, 224))
            image = Net.preprocess(image)  # ho zittito il Canny e la conversione a greyscale!!
            stato_erba = insert_tag(image)  # non tagliata :0
            occlusione = insert_occlusion(img_occlusion)  # restituisce 1 se c'è un ostacolo
            file_name = f'image_{x}_{y}.jpg'
            file_path = os.path.join(image_directory, file_name)
            cv2.imwrite(file_path, image_to_store)
            # Aggiorna la mappa con l'immagine e il tag
            mappa.update_map(x, y, image_to_store, stato_erba, occlusione)


def insert_tag(image):
    image = cv2.Canny(image, 10, 50)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)  # aggiungi channel_in
    model = Net.CNN()
    model.load_state_dict(torch.load("cnn.pth"))
    output = model(image)
    _, stato_erba = torch.max(output, dim=1)
    stato_erba = stato_erba.item()
    return stato_erba


def insert_occlusion(image):  # funzione per la depth estimation
    image_segm_input = image
    binary_mask, resized_image = create_object_mask(image_segm_input)
    areas = find_object_areas(binary_mask)

    if areas:
        depth_model = ANSRGB()
        state_dict = torch.load('ckpt.12.pth', map_location=torch.device('cpu'))
        depth_model.load_state_dict(state_dict, strict=False)
        depth_model.eval()
        img_model_input = depth_img_preprocessing(image, (3, 128, 128))
        with torch.no_grad():
            depth_output = depth_model({"rgb": img_model_input})

        depth_model_result = depth_output["occ_estimate"]

        depth_image = depth_model_result[:, 0, :, :].squeeze().numpy()
        average_distance = calculate_average_distance(depth_image, areas)
        print(f"Distanza media- funzione mappa: {average_distance:.3f}")
        if average_distance < 0.5:
            return 1
        else:
            return 0
    else:
        # print("non ci sono ostacoli")
        return 0


def trasforma_griglia(mappa):
    for row in range(mappa.grid.shape[0]):
        for col in range(mappa.grid.shape[1]):
            image = mappa.grid[row, col]
            if isinstance(image, np.ndarray):
                file_path = os.path.join(image_directory, f'image_{row}_{col}.jpg')
                cv2.imwrite(file_path, image)
                print(f"Immagine salvata in {file_path}")
                # Aggiorna la griglia con il percorso del file
                mappa.grid[row, col] = file_path
                print(type(mappa.grid[row, col]))


def save_tags_to_json(mappa, json_file):
    tags = {}
    for i in range(mappa.grid.shape[1]):
        for j in range(mappa.grid.shape[0]):
            tag = mappa.tags[i, j]
            tags[f"{i},{j}"] = tag
    with open(json_file, 'w') as f:
        json.dump(tags, f)

    print("tags salvati in {json_file}")


def save_occlusions_to_json(mappa, json_file):
    occulsions = {}
    for i in range(mappa.grid.shape[1]):
        for j in range(mappa.grid.shape[0]):
            occlusion = mappa.occlusion_grid[i, j]
            occulsions[f"{i},{j}"] = occlusion
    with open(json_file, 'w') as f:
        json.dump(occulsions, f)

    print("occlusioni salvate in {json_file}")


def save_image_paths_to_json(mappa, json_file):
    # Dizionario per memorizzare i percorsi delle immagini con le loro posizioni
    image_paths = {}
    for i in range(mappa.grid.shape[1]):
        for j in range(mappa.grid.shape[0]):
            image = mappa.grid[i, j]
            if isinstance(image, str):  # Verifica se l'elemento è un percorso (stringa)
                image_paths[f"{i},{j}"] = image  # Salva la posizione e il percorso               !!ho flippato

    # Scrivi il dizionario nel file JSON
    with open(json_file, 'w') as f:
        json.dump(image_paths, f)

    print(f"Percorsi delle immagini salvati in {json_file}")


def load_tags_from_json(mappa, json_file):
    with open(json_file, 'r') as f:
        tags = json.load(f)  # Carica i dati dal file JSON

    # Assegna i tag caricati alla matrice mappa.tags
    for key, value in tags.items():
        i, j = map(int, key.split(','))  # Converte la chiave "i,j" in due interi
        mappa.tags[i, j] = value  # Assegna il valore al tag corrispondente
        print(f"Caricato tag {value} in posizione ({i}, {j})")  # Verifica visiva
    print(mappa.tags)  # Stampa finale per vedere l'intera matrice


def load_occlusions_from_json(mappa, json_file):
    with open(json_file, 'r') as f:
        occlusions = json.load(f)  # Carica i dati dal file JSON

    # Assegna i tag caricati alla matrice mappa.tags
    for key, value in occlusions.items():
        i, j = map(int, key.split(','))  # Converte la chiave "i,j" in due interi
        mappa.occlusion_grid[i, j] = value  # Assegna il valore al tag corrispondente
        print(f"Caricata occlusione {value} in posizione ({i}, {j})")  # Verifica visiva
    print(mappa.occlusion_grid)  # Stampa finale per vedere l'intera matrice


def load_image_paths_from_json(mappa, json_file):
    # Leggi il file JSON
    with open(json_file, 'r') as f:
        image_paths = json.load(f)

    # Scorri il dizionario e reinserisci i percorsi nella griglia
    for key, value in image_paths.items():
        i, j = map(int, key.split(','))  # Converte "i,j" in due interi
        mappa.grid[i, j] = value  # Reinserisce il percorso nella griglia

    print(f"Percorsi delle immagini caricati da {json_file}")


'''



image_files = [file for file in os.listdir(c.FOLDER_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]
mappa=mp.Map(c.MAP_WIDTH,c.MAP_HEIGHT,c.CELL_SIZE)
memory_path = 'C:\\CVCS\\memoria'
image_directory = os.path.join(memory_path, 'immagini_mappa')
json_file= os.path.join(memory_path, 'image_paths.json')

riempi_mappa(mappa)
trasforma_griglia(mappa)
save_image_paths_to_json(mappa, json_file)
load_image_paths_from_json(mappa, json_file)

mappa.print_tag_grid()
mappa.display_map()
'''
