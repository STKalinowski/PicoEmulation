import csv
import random
import pickle
import cv2
import torch
import os
import math


# given path to video, load and then turn into array of numpy frames
def loadVidAndFrame(path):
  cap = cv2.VideoCapture(path)
  frames = []
  while True:
    ret,frame = cap.read()
    if not ret:
      break
    #cv2_imshow(frame)
    frame = torch.from_numpy(frame).permute(2, 0, 1)
    frames.append(frame)
  cap.release()
  return frames

# Convert my string of lists back to lists for controler input
# A little hack for now
def parseList(inpStr):
  retVal = []
  holdList = []
  val = ''
  inListFlag = False
  for i in inpStr[1:-1]:
    if i == '[':
      holdList = []
      inListFlag = True
    elif i == ']':
      if val != '':
        holdList += [int(val)]
        val = ''
      retVal.append(holdList)
      inListFlag = False
    elif i == ',' and inListFlag:
      holdList += [int(val)]
      val = ''
    elif inListFlag:
      val += i
  return retVal

# Given the inputs and the frames
# Create data entries for a window size of x
def formatData(inputs, frames, nGram):
  retVal = []
  inputs = parseList(inputs)
  inputs = [[]]*nGram + inputs
  inputs = [torch.zeros((1,6)).scatter_(1, torch.tensor([x]), 1) for x in inputs]
  frames = [torch.zeros((3,128,128))]*nGram + [frame for frame in frames] 
  frames = [frame/127.5 - 1 for frame in frames]
  for i in range(len(frames)-nGram):
    xFrames = torch.stack(frames[i:i+nGram], dim=1)
    xInputs = torch.cat(inputs[i:i+nGram], dim=0)
    yFrame = frames[i+nGram]
    retVal += [((xFrames, xInputs), yFrame)]
  return retVal

def main(runId):
    # Read in and split the data accordingly into the different sections of val, train, test
    # By labeling the rows instead of processed tensors, we can reconfigure and change in the future.
    trainFile = open(f'./Train/{runId}.csv', 'w')
    trainWriter = csv.writer(trainFile,delimiter=',')
    valFile = open(f'./Val/{runId}.csv', 'w')
    valWriter = csv.writer(valFile, delimiter=',')
    testFile = open(f'./Test/{runId}.csv', 'w')
    testWriter = csv.writer(testFile, delimiter=',')

    # Train, Val, Test
    splits = [0.8,0.1,0.1]
    splitType = ['train', 'val', 'test']
    writer = {'train':trainWriter, 'val':valWriter, 'test':testWriter}
    loss = torch.nn.functional.mse_loss
    NGRAM = 4

    # Create Frame and Control Tensors
    with open('./HoldRes/'+runId+'.csv', 'r') as csvF:
        reader = csv.reader(csvF, delimiter=',')
        next(reader)
        for row in reader:
            assigned = random.choices(splitType, weights=splits)[0]
            # Read Frame
            frames = loadVidAndFrame(f'./HoldRes/{row[0]}.mp4') 
            formated = formatData(row[1], frames, NGRAM) 
            # Write tensor + row in file
            for i in range(len(formated)):
                with open(f'./{assigned}/{row[0]}{i}.pickle', 'wb') as tF:
                    p = pickle.dumps(formated[i])
                    tF.write(p)
                l = float(loss(formated[i][0][0][:,-1,:,:], formated[i][1]))
                writer[assigned].writerow(row + [str(row[0])+str(i)+'.pickle', str(l)])
    trainFile.close()
    valFile.close()
    testFile.close()

    files = {'train':f'./Train/{runId}.csv', 'val':f'./Val/{runId}.csv', 'test':f'./Test/{runId}.csv'}
    # Calculate ZScore and add 
    for i in splitType:
        # Get Means
        sumL = 0
        total = 0
        mean = 0
        with open(files[i], 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                sumL += float(row[-1])
                total += 1
        if total > 0:
            mean = sumL / total
        # Get SD
        diff = 0
        sd = 0
        with open(files[i], 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                diff += (float(row[-1])  - mean)**2
        if total > 0:
            sd = (diff / total)**(0.5)

        # Label and Create ZScore
        with open(files[i], 'r') as f:
            with open(files[i][:-4]+'tmp.csv', 'w') as tmpF:
                reader = csv.reader(f, delimiter=',')
                writer = csv.writer(tmpF, delimiter=',')
                for row in reader:
                    zScore = abs(math.floor( (float(row[-1]) - mean) / sd))
                    writer.writerow(row + [zScore])
        os.remove(files[i])
        os.rename(files[i][:-4]+'tmp.csv', files[i])

if __name__ == '__main__':
    # Args
    main('2305180056')