import numpy as np
import random
import time
import constants as c
import map
import cv2
import os


def random_img():  # restituisce un'immagine random
    image_files = [file for file in os.listdir(c.FOLDER_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]
    random_image = random.choice(image_files)
    example_image = cv2.imread(f"{c.FOLDER_PATH}" + random_image)
    example_image = cv2.resize(example_image, c.CELL_SIZE)  # Ridimensiona l'immagine alla dimensione della cella

    return example_image


def streamin():  # generatore dati
    vettore = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    a = 0
    b = 0
    c = 0
    for i in range(8, 15):
        bit = random.randint(0, 1)
        vettore[i] = bit

    for i in range(0, 8):
        a += vettore[7 - i] * 2 ** i

    for i in range(8, 16):
        b += vettore[23 - i] * 2 ** (i - 8)

    for i in range(16, 24):
        c += vettore[39 - i] * 2 ** (i - 16)

    return a, b, c


def processData(data):  # converte direzione e misura
    sens = 0

    # rimuove il prefisso "0b" e assicura che la rappresentazione
    # abbia una lunghezza di 8 bit riempiendo con zeri a sinistra
    bit_representation = bin(data)[2:].zfill(8)
    add_bit = bit_representation[0:2].zfill(8) # due bit per direzione
    measure_bit = bit_representation[2:].zfill(8)  # sei bit per misurazione

    if (int(add_bit, 2) == 0): sens = 1  # left
    if (int(add_bit, 2) == 1): sens = 2  # front
    if (int(add_bit, 2) == 2): sens = 3  # right
    if (int(add_bit, 2) == 3): sens = 4  # back

    measure = int(measure_bit, 2)

    return sens, measure


if __name__ == "__main__":

    # inizializza la posizione del robot al centro
    # todo da calcorare in modo automatico date le dimensioni della matrice
    pos_x = 2
    pos_y = 2

    matrix_map = map.Map(c.MAP_WIDTH, c.MAP_HEIGHT, c.CELL_SIZE)

    cont = 10  # contatore per evitare ciclo infinito

    while cont > 0:  # sostituire cont>0 con True per ciclo infinito
        start_time = time.time()
        time.sleep(2)
        timer = time.time()
        cont -= 1  # commentare per ciclo infinito

        while timer >= 2:
            timer = 0
            pacchetto_dati = streamin()

            if pacchetto_dati[0] == 255 and pacchetto_dati[2] == 254:
                print(processData(pacchetto_dati[1]))
                byte_utile = processData(pacchetto_dati[1])
                sens = byte_utile[0]  # direzione
                measure = byte_utile[1]  # distanza

                if byte_utile[0] == 2:  # front
                    pos_y -= 1
                    matrix_map.update_map(pos_x,pos_y,random_img())

                elif byte_utile[0] == 4:  # back
                    pos_y += 1
                    matrix_map.update_map(pos_x, pos_y, random_img())

                elif byte_utile[0] == 1:  # left
                    pos_x -= 1
                    matrix_map.update_map(pos_x, pos_y, random_img())

                elif byte_utile[0] == 3:  # right
                    pos_x += 1  # passo 1 da sostituire con measure
                    matrix_map.update_map(pos_x, pos_y, random_img())

                # condizioni per allargare la matrice
                if pos_x <= (len(matrix_map.grid) - (len(matrix_map.grid)-1)) or pos_x >= (len(matrix_map.grid) - 2):
                    print("oltre indice 1")
                    print(pos_x, pos_y)
                    print(len(matrix_map.grid), len(matrix_map.grid[0]))
                    print("ingrandimento")
                    matrix_map.ingrandisci_matrice()
                    print(len(matrix_map.grid), len(matrix_map.grid[0]))
                    pos_x += 2
                    pos_y += 2
                    print(pos_x, pos_y)

                if pos_y <= (len(matrix_map.grid[0]) - (len(matrix_map.grid[0])-1)) or pos_y >= (len(matrix_map.grid[0]) - 2):
                    print("oltre indice 2")
                    print(pos_x, pos_y)
                    print(len(matrix_map.grid), len(matrix_map.grid[0]))
                    print("ingrandimento")
                    matrix_map.ingrandisci_matrice()
                    print(len(matrix_map.grid), len(matrix_map.grid[0]))
                    pos_x += 2
                    pos_y += 2

            print(pos_x, pos_y)

        matrix_map.display_grid()

    matrix_map.display_map()







