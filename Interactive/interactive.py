import cv2
import numpy as np
import time
from pynput import keyboard
import threading
import mss 
import pickle
import torch
import gameModel

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

def createInputTensor(input):
    input = parseList(list(input))
    retVal = torch.zeros((1,6)).scatter_(1, torch.tensor([input]), 1)
    return retVal 

#######
key_set = set()
window_pressed = set()
key_index_map = {'Key.up': 0, 'Key.down': 1, 'Key.left': 2, 'Key.right': 3, "'c'": 4, "'x'": 5}
lock = threading.Lock()
displayFrame = None

def update_one_hot_set(key, value):
    with lock:
        # If the pressed/release key is one of the keys we're interested in, update the one hot set
        if key in key_index_map:
            if value:
                key_set.add(key_index_map[key])
                window_pressed.add(key_index_map[key])
            else:
                key_set.discard(key_index_map[key])


def on_press(key):
    try:
        update_one_hot_set(str(key), True)
    except AttributeError:
        # Ignore spcecial keys
        pass

def on_release(key):
    try:
        update_one_hot_set(str(key), False)
    except AttributeError:
        # Ignore special keys
        pass

def cvDisplay(fps):
    global displayFrame
    frame_box = 128
    cv2.namedWindow('Game', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Game', frame_box, frame_box)
    delay = int(1000/fps)
    while True:
        print(displayFrame)
        print(displayFrame.shape)
        print(type(displayFrame))
        cv2.imshow('Game', displayFrame)
        cv2.waitKey(delay)

# Game loop, update frames with output from model.
def updateFrame(startFrame):
    global window_pressed
    global key_set
    global displayFrame
    fps = 0.1
    sleepTime = 1/fps
    with open(f'./{startFrame}.pickle', 'rb') as f:
        beginT = pickle.load(f)
    frames = beginT[0][0]
    inputs = beginT[0][1]
    tmpCtrl = None
    norm_layer = torch.nn.LayerNorm
    config={
        "epoch":5,
        "batch_size":8,
        "lr":0.0001,
        "img_size":128,
        "patch_size":16,
        "in_chans":3,
        "embed_dim":768,
        "depth":3,
        "num_heads":4,
        "mlp_ratio":4.,
        "qkv_bias":False, 
        "qk_scale":None, 
        "drop_rate":0.,
        "attn_drop_rate":0.,
        "drop_path_rate":0.1, 
        "hybrid_backbone":None,
        "norm_layer":norm_layer, 
        "num_frames":4, 
        "dropout":0.}
    model = gameModel.GameEmulator(img_size=config['img_size'],patch_size=config['patch_size'], in_chans=config['in_chans'], embed_dim=config['embed_dim'], depth=config['depth'], num_heads=config['num_heads'],
                     mlp_ratio=config['mlp_ratio'], qkv_bias=config['qkv_bias'], qk_scale=config['qk_scale'],drop_rate=config['drop_rate'],attn_drop_rate=config['attn_drop_rate'],
                     drop_path_rate=config['drop_path_rate'], hybrid_backbone=config['hybrid_backbone'], norm_layer=norm_layer, num_frames=config['num_frames'], dropout=config['dropout'])
    model.eval()

    currentFrame = frames[:,-1,:,:]
    currentFrame = (currentFrame.permute(1,2,0).numpy()+1)*127.5 
    currentFrame = cv2.convertScaleAbs(currentFrame)
    displayFrame = currentFrame
    displayThread = threading.Thread(target=cvDisplay, args=(fps,))
    displayThread.start()

    # 3 4 128 128
    # CV uses H,W,C!
    #currentFrame = frames[:,-1, :,:]
    while(True):
        #currentFrame = (currentFrame.permute(1,2,0).numpy()+1)*127.5
        currentFrame = cv2.convertScaleAbs(currentFrame)
        displayFrame = currentFrame

        # The controls captured in another thread in the background, just wake up, get controls and produce the next frame
        with lock:
            tmpCtrl = window_pressed
            window_pressed = key_set

        tmpCtrl = createInputTensor(tmpCtrl)
        inputs = torch.cat((inputs[1:,:], tmpCtrl), 0)
        output = model.forward(frames.unsqueeze(0), inputs.unsqueeze(0))
        output = output[0,-1,:,:,:]
        frames = torch.cat( (frames[:,1:,:,:],output.view(3,1,128,128)), 1)
        currentFrame = ((output.permute(1,2,0)+1)*127.5).detach()
        currentFrame = currentFrame.numpy()


        timeToWake = time.perf_counter() + sleepTime
        while time.perf_counter() < timeToWake:
            time.sleep(0)
    displayThread.join()

inp = None#input('Test or Train\n')
if inp != 'Train':
    inp = 'Test'                    
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    updateFrame(inp)
    cv2.destroyAllWindows()
    listener.stop()
    listener.join()