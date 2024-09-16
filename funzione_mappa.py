import os
import numpy as np
import constants as c
import map as mp
import random
import  cv2
import torch
import json
import Rete as Net


image_files = [file for file in os.listdir(c.FOLDER_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]
memory_path = 'C:\\CVCS\\memoria'
image_directory = os.path.join(memory_path, 'immagini_mappa')



def new_image(image_files):
    
    if not image_files:
        print("Nessun file immagine trovato nel percorso specificato.")
        return None

    random_image = random.choice(image_files)
    full_image_path = os.path.join(c.FOLDER_PATH, random_image)
    example_image = cv2.imread(full_image_path)
    return example_image

def riempi_mappa(mappa):
    for y in range (c.MAP_HEIGHT):
            #condizione ingrandimento
            for x in range (c.MAP_WIDTH):
            #condizione ingrandimento
                image = new_image(image_files)
                image_to_store = cv2.resize(image, (224,224))
                image=Net.preprocess(image) #ho zittito il Canny e la conversione a greyscale!!
                stato_erba=insert_tag(image) #non tagliata :0
                file_name = f'image_{x}_{y}.jpg'
                file_path = os.path.join(image_directory, file_name)
                cv2.imwrite(file_path, image_to_store)
                # Aggiorna la mappa con l'immagine e il tag
                mappa.update_map(x, y, image_to_store, tag=stato_erba)
                


def insert_tag(image):
   
    image = cv2.Canny(image, 10, 50)
    image=torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0) #aggiungi channel_in
    model = Net.CNN() 
    model.load_state_dict(torch.load("cnn.pth"))
    output=model(image)
    _, stato_erba = torch.max(output, dim=1)
    stato_erba=stato_erba.item()
    return stato_erba



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
                    print(type(mappa.grid[row,col]))

    


def save_image_paths_to_json(mappa, json_file):
    # Dizionario per memorizzare i percorsi delle immagini con le loro posizioni
    image_paths = {}

    # Scorri la griglia e salva i percorsi
    for i in range(mappa.grid.shape[1]):
        for j in range(mappa.grid.shape[0]):
            image = mappa.grid[i, j]
            if isinstance(image, str):  # Verifica se l'elemento è un percorso (stringa)
                image_paths[f"{i},{j}"] = image  # Salva la posizione e il percorso               !!ho flippato

    # Scrivi il dizionario nel file JSON
    with open(json_file, 'w') as f:
        json.dump(image_paths, f)

    print(f"Percorsi delle immagini salvati in {json_file}")

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