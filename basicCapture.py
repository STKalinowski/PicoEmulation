import cv2
import numpy as np
import time
import datetime
import mss
from pynput import keyboard
import threading
import pandas as pd

key_set = set()
window_pressed = set()
key_index_map = {'Key.up': 0, 'Key.down': 1, 'Key.left': 2, 'Key.right': 3, "'c'": 4, "'x'": 5}
# Define a lock to synchronize access to the one hot set
lock = threading.Lock()

# Keyboard functions
def update_one_hot_set(key, value):
    with lock:
        # If the pressed/released key is one of the keys we're interested in, update the one hot set
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
        # Ignore special keys
        pass

def on_release(key):
    try:
        update_one_hot_set(str(key), False)
    except AttributeError:
        # Ignore special keys
        pass

# Record frames & get Keys
def record():
    game = 'celest'
    global window_pressed
    global key_set
    # Define the screen size and region to capture
    screen_size = {"top":105, "left":1262, "width":128, "height":128}
    #r = (0, 0, screen_size[0], screen_size[1])
    imgReshape = (128,128)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    df = pd.DataFrame(columns=['id', 'keys', 'game'])
    count = 0
    totalTime = 3 # Total Time in seconds
    fps = 15 # Rate of frame capture in frames per second
    numOfLoops = totalTime * fps
    sleepTime = 1 / fps
    runs = 100
    with mss.mss() as sct:
        for i in range(runs):
            recordFrame = []
            recordKey = []
            for count in range(numOfLoops):
                # Capture a frame from the screen
                with lock:
                    sct_img = np.array(sct.grab(screen_size))
                    recordFrame.append(sct_img)
                    if len(window_pressed) > 0: 
                        recordKey.append(list(window_pressed))
                    else:
                        recordKey.append([])
                    window_pressed = key_set

                timeToWake = time.perf_counter() + sleepTime
                while time.perf_counter() < timeToWake:
                    time.sleep(0) 

            now = datetime.datetime.now()
            recordDate = now.strftime("%y%m%d%H%S")
            vid = cv2.VideoWriter(f'./HoldRes/{recordDate}.mp4', fourcc, float(fps), imgReshape)

            for frame in recordFrame:
                img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                img = cv2.resize(img, imgReshape)
                vid.write(img)
            df.loc[len(df)] = [recordDate, recordKey, game]

            time.sleep(2)

    df.to_csv(f'./HoldRes/{recordDate}.csv', index=False)


with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    record()
    listener.stop()
    listener.join()

