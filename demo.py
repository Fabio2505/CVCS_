import numpy as np
import torch
import map as mp
import funzione_mappa as fp
import constants as c
import retrieval_CV as ret
import Rete as Net
import os
import cv2
from torchvision import models, transforms
from PIL import Image





def show_image(image):
   if image is not None:
    # Visualizza l'immagine (opzionale)
       cv2.imshow('Random Image', image)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
       
   else:
      print("Non è stata trovata o caricata nessuna immagine.")
   return None


def insert_tag(image):
   
   image = cv2.Canny(image, 10, 50)
   image=torch.tensor(image, dtype=torch.float32)
   image = image.unsqueeze(0) #aggiungi channel_in
   output=model(image)
   _, stato_erba = torch.max(output, dim=1)
   stato_erba=stato_erba.item()
   return stato_erba


def controlla_giardino(mappa):
   giardino_noto=0
   sample=ret.new_image(image_files)  
   Resnet=ret.istanzia_modello()
   
   if isinstance(sample, np.ndarray):
      sample = Image.fromarray(sample)
   
   transform = transforms.Compose([
      transforms.Resize((224, 224)),  # Dimensione richiesta da ResNet
      transforms.ToTensor(),  # Converti in tensore PyTorch
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizzazione consigliata ImageNet
   ])
   sample=transform(sample)
   immagini=mappa.estrai_foto()

   for idx, immagine in enumerate(immagini):
      immagini[idx] = ret.extract_features(transform(immagine),Resnet)
      if(ret.compare_images(sample,immagini[idx])>0.8):
         giardino_noto=1

   return giardino_noto   



#CORE


image_files = [file for file in os.listdir(c.FOLDER_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]
mappa=mp.Map(c.MAP_WIDTH,c.MAP_HEIGHT,c.CELL_SIZE)
model = Net.CNN() 
model.load_state_dict(torch.load("cnn.pth"))
memory_path = 'C:\\CVCS\\memoria'
image_directory = os.path.join(memory_path, 'immagini_mappa')
json_file= os.path.join(memory_path, 'image_paths.json')
json_tags_file= os.path.join(memory_path, 'tags_file.json')



# RETRIEVAL
# se la mappa c'è già:
# Tentativo di caricamento della mappa dal file

if (os.path.isfile(json_file)):  # Verifica che 'json_file' esista ed è un file
      # Carica la mappa dal file
      fp.load_image_paths_from_json(mappa, json_file)
      fp.load_tags_from_json(mappa,json_tags_file)
      print('percorso caricato')
      
      
else:  #questo branch non gestisce i tag, si implementa con un json come per le img ma è più facile
   print("Il file json non esiste.")
   mappa=mp.Map(c.MAP_WIDTH,c.MAP_HEIGHT,c.CELL_SIZE)
   fp.riempi_mappa(mappa) #scrive i percorsi nella mappa e salva le foto nella cartella
   fp.trasforma_griglia(mappa)
   fp.save_image_paths_to_json(mappa,json_file)
   fp.save_tags_to_json(mappa,json_tags_file)
   
   
mappa.display_map() #stampa i percorsi nella cartella concatenati
mappa.print_tag_grid()

print(f"MAP_WIDTH: {c.MAP_WIDTH}, MAP_HEIGHT: {c.MAP_HEIGHT}")



   #To Do: aggiungere depth, implementare navigazione.
   


   

