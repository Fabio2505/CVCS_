import numpy as np
import torch
import map as mp
import constants as c
import os
import cv2
from torchvision import models, transforms
import simulazione as sim
import retrieval_CV as ret
import Rete as Net

def show_image(image):
   if image is not None:
    # Visualizza l'immagine (opzionale)
       cv2.imshow('Random Image', image)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
       
       
       

       
   else:
      print("Non è stata trovata o caricata nessuna immagine.")
   return None


# RETRIEVAL

def controlla_giardino(mappa_giardino):
   giardino_noto=0
   sample=ret.new_image(image_files)  #nuova acquisizione
   Resnet=ret.istanzia_modello()

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

def insert_tag(image):
   
   image = cv2.Canny(image, 10, 50)
   image=torch.tensor(image, dtype=torch.float32)
   image = image.unsqueeze(0) #aggiungi channel_in
   output=model(image)
   _, stato_erba = torch.max(output, dim=1)
   stato_erba=stato_erba.item()
   return stato_erba

#CORE

image_files = [file for file in os.listdir(c.FOLDER_PATH) if file.endswith(('.jpg', '.jpeg', '.png'))]


mappa=mp.Map(c.MAP_WIDTH,c.MAP_HEIGHT,c.CELL_SIZE)
model = Net.CNN() 
model.load_state_dict(torch.load("cnn.pth"))



# se la mappa c'è già: puoi chiamare cotrolla giardino

#starting position
x=0
y=0
#comincia da in alto a destra ma si modifica con semplicità

for y in range (c.MAP_HEIGHT):
#condizione ingrandimento
   for x in range (c.MAP_WIDTH):
   #condizione ingrandimento
      image = ret.new_image(image_files)
      image_to_store = cv2.resize(image, (224,224))
      image=Net.preprocess(image) #ho zittito il Canny e la conversione a greyscale!!
      stato_erba=insert_tag(image) #non tagliata :0
      mappa.update_map(x,y,image_to_store,tag=stato_erba)
       




#missione: inserire tag nella casella della mappa
mappa.display_grid() 
mappa.display_map() #vuole l'RGB parlane con Giorgia 


#NAVIGAZIONE SIMPLE DA IMPLEMENTARE. #RETRIEVAL DA IMPLEMENTARE ED EVENTUALMENTE SFOLTIRE
