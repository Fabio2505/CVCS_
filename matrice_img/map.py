import numpy as np
import cv2
import matplotlib.pyplot as plt


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
        self.grid[y][x] = image
        self.occupancy_grid[y][x] = 1
        # self.tags[x][y] = {tag}

    # todo se aggiungo anche altre info, va ingrandita anche la tabella dei tag

    def ingrandisci_matrice(self):
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

    # Visualizza l'intera mappa con le immagini
    def display_map(self):
        # controlla se nella matrice ci sono celle vuote senza immagini
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] is None:
                    self.grid[i, j] = np.zeros((self.cell_size[0], self.cell_size[1], 3), dtype=np.uint8)  # inserisce Immagine nera
        """
        # Concatena le immagini lungo l'asse delle colonne per creare righe
        rows = [np.concatenate(self.grid[i, :], axis=1) for i in range(self.width)]

        # Concatena le righe per creare l'intera mappa
        map_image = np.concatenate(rows, axis=0)

        # Visualizza l'immagine della mappa
        cv2.imshow("Mappa", map_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        # Crea un'immagine che rappresenta l'intera mappa
        map_image = np.zeros((self.height * self.cell_size[0], self.width * self.cell_size[1], 3), dtype=np.uint8)

        # Riempie map_image con le immagini delle celle
        for i in range(self.height):
            for j in range(self.width):
                map_image[i * self.cell_size[0]: (i + 1) * self.cell_size[0], j * self.cell_size[1]: (j + 1) * self.cell_size[1]] = self.grid[i, j]

        cv2.namedWindow("map", cv2.WINDOW_NORMAL)
        ims = cv2.resize(map_image, (960, 960))  # resize dell'immagine intera per visualizzazione a schermo

        cv2.imshow('map', ims)
        cv2.waitKey(0)
        cv2.destroyAllWindows()







