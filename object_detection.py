import cv2
import numpy as np


def create_object_mask(image, target_size=(128, 128)):
    """
        Crea una maschera binaria per identificare oggetti nell'immagine.

        Questa funzione legge un'immagine da un percorso specificato, esegue un crop della parte superiore
        dell'immagine (50% dell'altezza), ridimensiona l'immagine risultante a una dimensione target,
        e infine applica una sogliatura per generare una maschera binaria degli oggetti.

        Argomenti:
            image :immagine da elaborare.
            target_size (tuple): Dimensioni desiderate per l'immagine ridimensionata (larghezza, altezza).
                                 Il valore predefinito è (128, 128).

        Ritorna:
            binary (numpy.ndarray): Maschera binaria dell'immagine, dove i pixel neri (0)) rappresentano
                                    gli oggetti identificati e i pixel bianchi (255) rappresentano il background.
            resized_cropped_image (numpy.ndarray): Immagine ridimensionata e croppata.
        """

    height, width, _ = image.shape
    cropped_image = image[:height // 2, :]  # Esegui un crop della parte superiore (50% dell'altezza)
    resized_cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # filtro gaussiano per ridurre il rumore
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)  # Usa la sogliatura per identificare i tronchi

    return binary, resized_cropped_image


def find_object_areas(binary_mask, min_area_threshold=450):
    """
    Trova le aree valide nella maschera binaria che superano la soglia min_area_threshold.

    Args:
        binary_mask (numpy.ndarray): Maschera binaria dei tronchi.
        min_area_threshold (int): Soglia minima per considerare un'area.

    Returns:
        List[numpy.ndarray]: Lista di maschere per le aree valide.
    """
    inverted_mask = cv2.bitwise_not(binary_mask)  # Inverti la maschera

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)

    valid_areas_mask = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area_threshold:
            # Crea una maschera per l'area specifica
            trunk_mask = (labels == label).astype(np.uint8) * 255
            valid_areas_mask.append(trunk_mask)

    return valid_areas_mask


def calculate_average_distance(depth_image, trunk_masks):
    """
    Calcola la distanza media basata sulle aree valide.

    Args:
        depth_image (numpy.ndarray): Immagine di profondità.
        trunk_masks (List[numpy.ndarray]): Lista di maschere.

    Returns:
        float: Distanza media distanza media più corta. Se non ci sono aree valide, restituisce 0.
    """
    distances = []

    for trunk_mask in trunk_masks:
        depth_values = depth_image[trunk_mask == 255]  # Estrai i valori di profondità

        # Se ci sono valori di profondità validi, calcola la distanza media
        if depth_values.size > 0:
            average_distance = np.mean(depth_values)
            distances.append(average_distance)

    return min(distances)




























