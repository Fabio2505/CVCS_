import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle 
import os
from PIL import Image
import json


class Map:
    def __init__(self, width, height, cell_size):
        self.width = width  # colonne
        self.height = height  # righe
        self.cell_size = cell_size
        self.grid = np.empty((height, width), dtype=object)  # Matrice per memorizzare le immagini
        self.tags = np.empty((height, width), dtype=object)  # Matrice per memorizzare info aggiuntive
        self.occupancy_grid = np.zeros((height, width))  # Matrice per test movimento robot e allargamento mappa automatico
        
    # Aggiorna la cella della mappa con l'immagine acquisita dal robot

    def update_map(self, x, y, image, tag=None):
        print(f"Update map at ({x}, {y})")
    
        self.grid[x, y] = image
        self.occupancy_grid[x, y] = 1  # Assicurati che sia 1, non un intero non subscriptable
        self.tags[x, y] = tag

        

    def ingrandisci_mappa(self):
        righe, colonne = len(self.grid), len(self.grid[0])
        self.height = 4 + righe  # 120+righe
        self.width = 4 + colonne  # 120+colonne

        nuova_matrice = np.empty((self.height, self.width), dtype=object)
        nuova_occupancy_grid = np.zeros((self.height, self.width))
        for i in range(righe):
            for j in range(colonne):
                nuova_matrice[i + 2][j + 2] = self.grid[i][j]  # i+60,j+60
                nuova_occupancy_grid[i + 2][j + 2] = self.occupancy_grid[i][j]

        self.grid = nuova_matrice
        self.occupancy_grid = nuova_occupancy_grid
    
    def estrai_foto(self):
        immagini = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] is not None:  # Verifica se c'è un'immagine nella cella
                    immagini.append(self.grid[i, j])  # Aggiungi l'immagine alla lista
        return immagini
    
    """
    # Mostra la singola immagine per cella (non utilizzato)
    def display_images(self):
        for x in range(self.width):
            for y in range(self.height):
                image = self.grid[x][y]
                tag_info = self.tags[x][y]
                print(tag_info)
                if image is not None:
                    cv2.imshow(f'Cella ({x}, {y})', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """

    # visualizza occupancy_grid
    def display_grid(self):
        plt.imshow(self.occupancy_grid, cmap='gray', origin='upper', extent=(0, self.width, self.height, 0))
        plt.show()

    

    def display_map(self):
        # Percorso del file JSON e della directory delle immagini
        memory_path = 'C:\\CVCS\\memoria'
        json_file = os.path.join(memory_path, 'image_paths.json')
        

        # Carica i percorsi delle immagini dal file JSON una sola volta
        with open(json_file, 'r') as f:
            image_paths = json.load(f)

        # Crea un'immagine vuota che rappresenta l'intera mappa
        map_image = np.zeros((self.height * self.cell_size[0], self.width * self.cell_size[1], 3), dtype=np.uint8)

        # Itera su ogni cella della griglia per caricare le immagini
        for i in range(self.height):
            for j in range(self.width):
                # Recupera il percorso dell'immagine dal JSON
                key = f"{i},{j}"
                file_path = image_paths.get(key)  # Ottieni il percorso dell'immagine dal JSON
                
                if file_path and os.path.exists(file_path):
                    image = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)

                    if image is not None:
                        # Inserisci l'immagine caricata nella griglia di mappa
                        map_image[i * self.cell_size[0]: (i + 1) * self.cell_size[0],
                                j * self.cell_size[1]: (j + 1) * self.cell_size[1]] = image
                    else:
                        print(f"Errore nel caricamento dell'immagine da {file_path}")
                else:
                    print(f"Percorso dell'immagine non valido o immagine mancante per ({i}, {j})")

        # Visualizza l'immagine della mappa
        cv2.namedWindow("Mappa", cv2.WINDOW_NORMAL)
        cv2.imshow("Mappa", map_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def print_tag_grid(self):
        # Crea una matrice vuota per i tag binari
        tag_grid = np.zeros((self.height, self.width))
        
        # Copia i valori dei tag solo nelle celle occupate
        for i in range(self.height):
            for j in range(self.width):
                if self.occupancy_grid[i][j] == 1 and self.tags[i][j] is not None:
                    tag_grid[i][j] = int(self.tags[i][j])

        # Definisci i colori per 0 e 1
        cmap = mcolors.ListedColormap(['white', 'green'])  # bianco per 0, verde per 1
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Crea la figura
        plt.figure(figsize=(self.width / 2, self.height / 2))
        plt.imshow(tag_grid, cmap=cmap, norm=norm)
        
        # Aggiungi una griglia per evidenziare le celle
        plt.grid(which='both', color='black', linestyle='-', linewidth=2)
        
        # Imposta le ticks per il numero corretto di righe e colonne
        plt.xticks(np.arange(-.5, self.width, 1), [])
        plt.yticks(np.arange(-.5, self.height, 1), [])
        
        # Mostra la mappa
        plt.show()
    
