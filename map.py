import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import json
import config_constants as c

#plt.switch_backend('MacOSX')


class Map:
    def __init__(self, width, height, cell_size):
        self.width = width  # colonne
        self.height = height  # righe
        self.cell_size = cell_size
        self.grid = np.empty((height, width), dtype=object)  # Matrice per memorizzare le immagini
        self.tags = np.empty((height, width), dtype=object)  # Matrice per memorizzare info aggiuntive
        self.nav_grid = []
        self.occlusion_grid = np.empty((height, width), dtype=object)  # Matrice per rilevare la presenza di un ostacolo
        self.path = []

    # Aggiorna la cella della mappa con l'immagine acquisita dal robot

    def update_map(self, x, y, image, tag, occlusion):
        print(f"Update map at ({x}, {y})")
        print(occlusion)
        self.grid[x, y] = image
        self.tags[x, y] = tag
        self.occlusion_grid[x, y] = occlusion

    """
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
    """

    def estrai_foto(self):
        immagini = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] is not None:  # Verifica se c'è un'immagine nella cella
                    immagini.append(self.grid[i, j])  # Aggiungi l'immagine alla lista
        return immagini

    def get_neighboring_values(self, i, j):
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                if self.tags[ni][nj] is not None:
                    neighbors.append(self.tags[ni][nj])
        return neighbors

    def create_nav_grid(self):
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if self.occlusion_grid[i][j] == 1:
                    row.append(1)  # Ostacolo
                elif self.tags[i][j] is None:  # cosidero il caso None solo per erba tagliata
                    neighbors = self.get_neighboring_values(i, j)
                    if neighbors.count(2) > neighbors.count(0):
                        row.append(2)
                    else:
                        row.append(0)
                elif self.tags[i][j] == 1:
                    row.append(2)  # Erba già tagliata
                else:
                    row.append(0)  # Erba non tagliata e libera
            self.nav_grid.append(row)

    def nav_path(self, start_pos):
        direzioni = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.path = []
        visitati = set()
        stack = []
        final_backtracking = []

        x, y = start_pos
        self.path.append((x, y))
        visitati.add((x, y))
        stack.append((x, y))

        def is_valid(nx, ny):
            # Verifica che la cella sia dentro i limiti e non sia un ostacolo o già visitata
            return (0 <= nx < len(self.nav_grid) and 0 <= ny < len(self.nav_grid[0]) and self.nav_grid[nx][
                ny] == 0 and (
                        nx, ny) not in visitati)

        def unvisited_cells():
            # Verifica se ci sono ancora celle visitabili nella matrice
            for i in range(len(self.nav_grid)):
                for j in range(len(self.nav_grid[0])):
                    if self.nav_grid[i][j] == 0 and (i, j) not in visitati:
                        return True
            return False

        while stack:
            trovato_prossima = False
            for dx, dy in direzioni:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny):
                    x, y = nx, ny
                    self.path.append((x, y))
                    visitati.add((x, y))
                    stack.append((x, y))
                    trovato_prossima = True
                    break

            if not trovato_prossima:
                if len(stack) > 1:
                    stack.pop()
                    x, y = stack[-1]

                    # Salva in percorso_backtracking solo se non ci sono più celle da visitare
                    if not unvisited_cells():
                        final_backtracking.append((x, y))
                    else:
                        self.path.append((x, y))
                else:
                    break

        return final_backtracking

    def display_map(self):
        # Percorso del file JSON e della directory delle immagini
        memory_path = c.HOME_PATH
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
                if self.tags[i][
                    j] is not None:  # qui ho tolto la condizione occupancy_grid[i][j]==1 che dava problemi in stampa
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

    def print_occlusion_grid(self):
        # Crea una matrice vuota per i tag binari
        occlusion_grid = np.zeros((self.height, self.width))

        # Copia i valori dei tag solo nelle celle occupate
        for i in range(self.height):
            for j in range(self.width):
                if self.occlusion_grid[i][
                    j] is not None:  # qui ho tolto la condizione occupancy_grid[i][j]==1 che dava problemi in stampa
                    occlusion_grid[i][j] = int(self.occlusion_grid[i][j])

        # Definisci i colori per 0 e 1
        cmap = mcolors.ListedColormap(['white', 'yellow'])  # bianco per 0, giallo per 1
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Crea la figura
        plt.figure(figsize=(self.width / 2, self.height / 2))
        plt.imshow(occlusion_grid, cmap=cmap, norm=norm)

        # Aggiungi una griglia per evidenziare le celle
        plt.grid(which='both', color='black', linestyle='-', linewidth=2)

        # Imposta le ticks per il numero corretto di righe e colonne
        plt.xticks(np.arange(-.5, self.width, 1), [])
        plt.yticks(np.arange(-.5, self.height, 1), [])

        # Mostra la mappa
        plt.show()

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    def display_path(self, start_pos):
        fig, ax = plt.subplots()
        matrice = np.array(self.nav_grid)

        # Creiamo una matrice per contare le visite nelle celle
        visite = np.zeros_like(matrice, dtype=int)

        # Conta le visite alle celle
        for (x, y) in self.path:
            visite[x, y] += 1

        # Mostra la matrice combinata con una mappa di colori
        cax = ax.matshow(matrice, cmap='Greys')

        # Disegna il percorso con una linea rossa
        x_coords, y_coords = zip(*self.path)
        ax.plot(y_coords, x_coords, color='red', linewidth=2, marker='o', markersize=5)

        # Cambia il colore delle celle visitate più di una volta
        for i in range(matrice.shape[0]):
            for j in range(matrice.shape[1]):
                if visite[i, j] > 1:
                    # Segnala le celle visitate più di una volta con un quadrato giallo
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='yellow', alpha=0.5))

        # Aggiungi le frecce per la direzione del percorso
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]

            # Disegna una freccia tra due celle successive
            ax.annotate('', xy=(y2, x2), xytext=(y1, x1),
                        arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))

        # Aggiungi annotazioni agli assi
        ax.set_xticks(np.arange(matrice.shape[1]))
        ax.set_yticks(np.arange(matrice.shape[0]))
        ax.set_xticklabels(np.arange(matrice.shape[1]))
        ax.set_yticklabels(np.arange(matrice.shape[0]))

        # Disabilita i bordi dei tick per una migliore visibilità
        ax.tick_params(top=False, bottom=False, left=False, right=False)

        # Aggiungi la posizione di partenza con una label
        start_x, start_y = start_pos
        ax.plot(start_y, start_x, color='green', marker='o', markersize=8)  # Mostra la posizione di partenza
        ax.text(start_y, start_x, f'Start\n({start_x}, {start_y})', color='green', fontsize=10, ha='right',
                va='bottom')  # Etichetta della posizione di partenza

        # Crea una legenda per spiegare il significato dei colori
        legend_elements = [
            mpatches.Rectangle((0, 0), 1, 1, facecolor='yellow', edgecolor='yellow', alpha=0.5,
                               label='Visited multiple times'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor='black', edgecolor='black', label='Obstacle'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='gray', label='Cut grass area')
        ]

        # Aggiungi la legenda al grafico
        ax.legend(handles=legend_elements, loc='best', fontsize=10)

        plt.show()




