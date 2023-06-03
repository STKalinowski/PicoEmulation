import tkinter as tk
from PIL import Image,ImageTk
import gameModel
import threading
import torch
from torch import nn
import pickle
import time

# Create the GUI window
window = tk.Tk()
window.title("Real-time Image Display")

# Create a canvas to display the images
canvas = tk.Canvas(window, width=128, height=128)
canvas.pack()
window_pressed = set()
key_set = set()
key_index_map = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3, "c": 4, "x": 5}
lock = threading.Lock()

class ImageGenerator():
    def __init__(self,config, StartFrames=None, StartInputs=None):
        self.model = gameModel.GameEmulator(img_size=config['img_size'],patch_size=config['patch_size'], in_chans=config['in_chans'], embed_dim=config['embed_dim'], depth=config['depth'], num_heads=config['num_heads'],
                     mlp_ratio=config['mlp_ratio'], qkv_bias=config['qkv_bias'], qk_scale=config['qk_scale'],drop_rate=config['drop_rate'],attn_drop_rate=config['attn_drop_rate'],
                     drop_path_rate=config['drop_path_rate'], hybrid_backbone=config['hybrid_backbone'], norm_layer=nn.LayerNorm, num_frames=config['num_frames'], dropout=config['dropout'])
        self.model.eval()
        
        self.frameTensors = StartFrames
        if self.frameTensors == None:
            # Need to make empty tensor Shape [1 3 config 128 128] <- 128 shoudl be config too?
            self.frameTensors = torch.zeros((1, 3, config['in_chans'],config['img_size'], config['img_size']))
        self.inputTensors = StartInputs
        if self.inputTensors == None:
            # Need shape [1, config.inputs?]
            self.inputTensors = torch.zeros((1,6))

    def parseList(self, inpStr):
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

    def createInputTensor(self, input):
        input = self.parseList(list(input))
        retVal = torch.zeros((1,6)).scatter_(1, torch.tensor([input]), 1)
        return retVal

    def generateNextImage(self):
        global window_pressed
        global key_set
        # Create input tensor and reset
        with lock:
            tmpCtrl = window_pressed
            window_pressed = key_set
        print(f'tmpCtrl: {tmpCtrl}')
        newInpTens = self.createInputTensor(tmpCtrl)
        self.inputTensors = torch.cat((self.inputTensors[1:,:], newInpTens),0)

        # Create image 
        output = self.model.forward(self.frameTensors.unsqueeze(0), self.inputTensors.unsqueeze(0))
        output = output[0, -1, :,:,:]
        self.frameTensors = torch.cat((self.frameTensors[:,1:,:,:], output.view(3,1,128,128)),1)

        img = ((output.permute(1,2,0)+1)*127.5).detach()
        img = img.numpy().astype('uint8')
        return img

    def load_state_dict(self,weights):
        self.model.load_state_dict(weights)

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
        "norm_layer":nn.LayerNorm, 
        "num_frames":4, 
        "dropout":0.}
with open(f'./Train.pickle', 'rb') as f:
    beginT = pickle.load(f)
stFrames = beginT[0][0]
stInputs = beginT[0][1]
imgModel = ImageGenerator(config, StartFrames=stFrames, StartInputs=stInputs)
imgModel.load_state_dict(torch.load('./GameEmulator.pt', map_location=torch.device('cpu')))

# Define a function to generate images based on user controls
def generate_image(canvas):
    startTime = time.time()
    image = imgModel.generateNextImage()
    endTime = time.time()
    elapsed = endTime - startTime
    print(f'Time take: {elapsed:.6f}')

    image = image[:, :, ::-1]
    image = Image.fromarray(image)
    canvas.image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=canvas.image_tk)
    window.after(100, generate_image, canvas)  # Adjust the delay (in ms) as needed


def update_one_hot_set(key, value):
    print(f'Key: {key}, Value: {value}')

    with lock:
        if key in key_index_map:
            if value:
                key_set.add(key_index_map[key])
                # Window pressed is only set not cleared
                # Clearing happens after image generation
                # This is because of the way we capture frames
                # We want all inputs within the img capture window!
                window_pressed.add(key_index_map[key])
            else:
                key_set.discard(key_index_map[key])

def handleKeyPress(key):
    try:
        update_one_hot_set(str(key), True)
    except AttributeError:
        pass

def handleKeyRelease(key):
    try:
        update_one_hot_set(str(key), False)
    except AttributeError:
        pass

# Bind arrow keys to capture user controls
def on_key_press(event):
    handleKeyPress(event.keysym)
def on_key_release(event):
    handleKeyRelease(event.keysym)

# Bind the key press event to the window
window.bind('<KeyPress>', on_key_press)
window.bind('<KeyRelease>', on_key_release)
# Start the image generation and display
generate_image(canvas)
# Start the Tkinter event loop
window.mainloop()
