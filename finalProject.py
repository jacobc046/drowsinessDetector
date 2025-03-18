import cv2
import numpy as np
import time

import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from PIL import Image

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 12, 7) #3 input channels, num feature maps, 5x5 kernel (filter). this leaves 12 24x24 featrure maps
        self.conv2 = nn.Conv2d(12, 24, 5) # (24, 30, 30)
        self.conv3 = nn.Conv2d(24, 48, 3)  # (48, 13, 13)
        self.pool = nn.MaxPool2d(2,2) #2x2 max pool
        
        self.fc1 = nn.Linear(48 * 6 * 6, 120)  # flattened 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
    def forward(self, x):
        #hidden 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        #hidden 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        #hidden 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x

net = NeuralNet()
net.load_state_dict(torch.load('./trained_net.pth', map_location=torch.device('cpu')))
net.eval()

class_names = ['closed eyes', 'open eyes']

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

liveVideo = cv2.VideoCapture(0)

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
lineThickness = 2

frame_count = 250
prediction = ""
while True:
    ret, img = liveVideo.read() # img at current instance, images in BGR form
    
    if not ret: #ensure frame is read properly
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert img to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 7) #detect faces within input using scale factor 1, min neighbors 5
    
    #detect face
    for (x,y,w,h) in faces:
        #define rectangle around face
        cv2.rectangle(img, (x,y), (x+w, y+h), blue, lineThickness)
        roi_gray = gray[y:y+h, x:x+w] #grayscale region of interest
        roi_color = img[y:y+h, x:x+w] #region of interest on colored image
        
        eyes = eye_cascade.detectMultiScale(roi_gray) #detect eyes within the recognized face
        for (ex, ey, ew, eh) in eyes:
            #inflate roi to include eyebrows
            ey -= 20
            eh += 10
            ew += 20
            ex -= 10
            if 140 > ew > 60 and 140 > eh > 60:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), green, lineThickness)
                input_image = roi_gray[ey: ey+eh, ex: ex+ew]
                if not input_image.any():
                    continue
                input_image = cv2.merge([input_image, input_image, input_image])
                input_image = cv2.resize(input_image, (72,72))
                
                input_image = torch.from_numpy(input_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    output = net(input_image)
                    _, predicted = torch.max(output, 1)
                    prediction = class_names[predicted.item()]
                    # print(prediction)
                # filename = f"captured_frame_{frame_count}.png"
                # cv2.imwrite(filename, input_image)
                # print(f"Saved {filename}")
                # frame_count += 1

        time.sleep(.07)
        
    #mirror image
    img = cv2.flip(img, 1)
    
    #add instruction to the screen
    height, width, _ = img.shape
    cv2.putText(img, "Press and hold esc to quit", (width//2,50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.3, color=red, thickness=lineThickness)
    cv2.putText(img, prediction, (width//2,150), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=red, thickness=lineThickness)
    
    #show output
    cv2.imshow('output', img)
    
    #if the escape key is pressed, stop the program (check every 1ms)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
#terminate program
liveVideo.release()
cv2.destroyAllWindows()