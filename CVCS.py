import torch 
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import os

batch_size = 16


# inizializzazione di un kernel customizzato per il filtraggio a sfocatura
customKernel = np.matrix([[0.1257861635, 0.0251572327, 0.03144654088, 0.251572327, 0.1257861635],
				[0.0251572327,0.05660377358, 0.07547169811,0.05660377358,0.0251572327],
				[0.03144654088, 0.07547169811, 0.09433962264,0.07547169811,0.03144654088],
				[0.0251572327,0.05660377358, 0.07547169811,0.05660377358,0.0251572327],
				[0.1257861635,0.0251572327,0.03144654088,0.251572327,0.1257861635]])

def gaussCustomFilter (sc) : #funzione per il filtraggio
	dst = cv2.filter2D(sc, -1, kernel = customKernel)
	#dst = cv2. GaussianBlur(sc, (3,3), 0)
	return dst

def custom_collate_fn(batch):
    # Estrai immagini ed etichette dal batch
    batch_images, batch_labels = zip(*batch)
    preprocessed_images, filtered_labels = custom_preprocess_and_filter(batch_images, batch_labels)
    
    return preprocessed_images, torch.tensor(filtered_labels, dtype=torch.long)

   
# 
 
def preprocess(image):
      
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        width, height = image.shape[:2]
        
        #crop
        start_width = max(int((width / 2) - 224), 0)  # Usa max per evitare valori negativi
        start_height = max(int((height / 2) - 112), 0)  # Usa max per evitare valori negativi

        cropped_image = image[start_height:start_height+224, start_width:start_width+448]
        canny_image = cv2.Canny(cropped_image, 10, 80)
       
        return canny_image
        

   
    

def custom_preprocess_and_filter(batch_images, batch_labels):
    preprocessed_batch = []
    filtered_labels = []
    i=0
    for image, label in zip(batch_images, batch_labels):
        i+=1
        image = preprocess(image)  # Assumi che 'preprocess' sia una tua funzione
        
         
        # Verifica se l'immagine processata soddisfa i tuoi criteri
        if (image.size > 0 and image.shape == (224, 448)):
            image=torch.tensor(image, dtype=torch.float32)
            image = image.unsqueeze(0) #aggiungi channel_in
            preprocessed_batch.append(image)
            filtered_labels.append(label)
            
        #else:
            
            #print("Immagine scartata, etichetta corrispondente scartata.")
            
            

    #pytorch
        preprocessed_batch_tensor = torch.stack(preprocessed_batch)
        filtered_labels_tensor = torch.tensor(filtered_labels)

    return preprocessed_batch_tensor, filtered_labels_tensor

        

class MyDataset(Dataset):

    def __init__(self, root_dir, class_label, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.file_names = os.listdir(root_dir)
        self.class_label = class_label  # Assegna l'etichetta di classe al dataset
        self.transform = transform  # Trasformazioni da applicare a ogni immagine

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_path).convert('RGB')

        # Applica le trasformazioni all'immagine, se specificate
        if self.transform:
            image = self.transform(image)

        label = self.class_label
        return image, label


dataset2 = MyDataset('C:\\Users\\racch\\OneDrive\\Desktop\\erbatagliata\\Erbalunga',0) #erba NON tagliata classe 0
dataset1 = MyDataset('C:\\Users\\racch\\OneDrive\\Desktop\\erbatagliata\\Erbacorta',1) #erba tagliata classe 1
print(len(dataset1))
print("Lunghezza 1 /n")
print(len(dataset2))
print("Lunghezza 2 /n")
dataset = ConcatDataset([dataset1, dataset2]) #unione dei due dataset in maniera non randomica.Quando li proponiamo alla rete randomizziamo l'indice senza ripetizioni
print("%  per dataset2") 
print( len(dataset2)/len(dataset))


total_size = len(dataset)
train_size = int(0.85 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


# Creazione dell'istanza del dataloader

num_epochs = 2 # EPOCHE per train

preprocessed_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

batch = next(iter(preprocessed_dataloader))
inputs = batch[0]
labels = batch[1]

print(batch[0].size())

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.dropout = nn.Dropout(0.4)  #con 0.3 fa 100% training set e 70 % test set!!!
        self.fc1 = nn.Linear(4 * 112 * 224, 16)
        self.fc2 = nn.Linear(16, 2)  # 2 output classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x)) 
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 4 * 224 * 112)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # nessuna funzione di attivazione dato che si usa la cross entropy
        return x
 



#Definizione della funzione di loss e del criterio di ottimizzazione
model = CNN() 
#model.load_state_dict(torch.load("cnn.pth"))
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) #0.01 standard

 
#TRAINING



for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (batch_images, batch_labels) in enumerate(preprocessed_dataloader):
        #print(inputs.size(), labels.size())  
    
       
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        

        labels = labels.long()
        loss = criterion(outputs, labels)

        # Backward pass e ottimizzazione
        loss.backward()
        optimizer.step()

        # Stampa le statistiche di allenamento
        running_loss += loss.item()
        if i % 50 == 0: # Stampa ogni i batch
            print('[Epoch %d, Batch %d] loss: %.2f' %
                  (epoch + 1, i + 1, running_loss/50 ))
            running_loss = 0.0

print('Training finito')
torch.save(model.state_dict(), "cnn.pth")

# TEST

model.load_state_dict(torch.load("cnn.pth"))
model.eval()


total_correct = 0
total_images = 0
with torch.no_grad():
    
    for images, labels in test_dataloader:
        

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_images += labels.size(0)
        total_correct += (predicted == labels).sum().item()


accuracy = total_correct / total_images

print(f'Accuracy on the test set: {accuracy:.2f}')





