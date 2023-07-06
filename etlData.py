import torch
from torchvision import transforms
import duckdb
import pandas as pd
import random
from diffusers import AutoencoderKL
import cv2
import time

# Process the videos into train,val,test parquets 
SPLIT = [0.8, 0.1, 0.1]
SPLITNAMES= ["Train", "Val", "Test"]
MODEL = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").encoder
CONTROL_SIZE = 6

def createOneHot(inputInt):
    retTensor = torch.zeros(6)
    pos = 5
    while inputInt > 0:
        retTensor[pos] = inputInt%2
        inputInt = inputInt >> 1
        pos -= 1
        assert pos >= -1
    return retTensor 

def createEntries(row):
    frames_encoded = []
    
    # Load & Encode the video, storing in frames_encoded
    video = cv2.VideoCapture('./Videos/'+row[2])
    success, image = video.read()
    transSequence = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transSequence(image).unsqueeze(0)
        frames_encoded.append(MODEL(image)[0].detach().numpy())
        # Read the next frame from the video
        success, image = video.read()
    video.release()

    # Create X & Y data
    x_data, y_data = [], []
    for i in range(len(frames_encoded) - 1):
        # For x data, we take the encoded frame and the one hot encoding of the input at the same index
        x = (frames_encoded[i], createOneHot(row[1][i]).numpy())
        # For y data, we take the next encoded frame
        y = frames_encoded[i + 1]
        
        x_data.append(x)
        y_data.append(y)
    
    # Convert the data to a DataFrame
    retDataframe = pd.DataFrame(list(zip(x_data, y_data)), columns=['x', 'y'])
    
    return retDataframe

def main():
    # Our starting parquets
    splitDfs = [pd.DataFrame(columns=['x', 'y']),pd.DataFrame(columns=['x', 'y']), pd.DataFrame(columns=['x', 'y'])]

    connection = duckdb.connect(database='recordings.db')
    result = connection.execute("SELECT * FROM recordings;")
    row = result.fetchone()
    count = 0
    while row is not None:
        start = time.time()
        idx = random.choices(range(len(splitDfs)),weights=SPLIT, k=1)[0]
        entries = createEntries(row)
        splitDfs[idx] = pd.concat([splitDfs[idx], entries])
        print(f'Count: {count}')
        end = time.time()
        print(f'Time: {end - start}')
        count +=1
        row = result.fetchone()


    # Save the dataframes
    for i in range(len(splitDfs)):
        fileName = SPLITNAMES[i]+'.parquet'
        splitDfs[i].to_parquet(fileName)
        

if __name__ == '__main__':
    main()