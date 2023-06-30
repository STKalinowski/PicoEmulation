import cv2
import numpy as np
import time
import datetime
import mss
from pynput import keyboard
import threading
import duckdb

key_index_map = {'Key.up': 1 << 0, 'Key.down': 1 << 1, 'Key.left': 1 << 2, 'Key.right': 1 << 3, "'c'": 1 << 4, "'x'": 1 << 5}
key_set = 0
window_pressed = 0
# Define a lock to synchronize access to the one hot set
lock = threading.Lock()

# Keyboard functions
def update_one_hot_set(key, value):
    with lock:
        # If the pressed/released key is one of the keys we're interested in, update the one hot set
        if key in key_index_map:
            key_set = key_set ^ key_index_map[key]
            if not(key_index_map[key] & window_pressed):
                window_pressed = key_index_map[key] ^ window_pressed

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
    connection = duckdb.connect(database='recordings.db')

    # Recording variables
    totalTime = 3 # Total Time in seconds
    fps = 15 # Rate of frame capture in frames per second
    numOfLoops = totalTime * fps
    sleepTime = 1 / fps
    runs = 100

    # Define the screen size and region to capture
    screen_size = {"top":105, "left":1262, "width":128, "height":128}
    imgReshape = (128,128)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    with mss.mss() as sct:
        for i in range(runs):
            recordFrame = []
            recordKey = []
            # One Recording, where we get a list of inputs and frames
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

            # Now we take our inputs add them to the DB and save our video 
            now = datetime.datetime.now()
            recordDate = now.strftime("%y%m%d%H%S")
            vid = cv2.VideoWriter(f'./Videos/{recordDate}.mp4', fourcc, float(fps), imgReshape)
            for frame in recordFrame:
                img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                img = cv2.resize(img, imgReshape)
                vid.write(img)
            # Write our control recording to our databases
            connection.execute("INSERT INTO recordings VALUE (?, ?, ?, ?)", [int(recordDate), recordKey,recordDate+'.mp4', game])
            time.sleep(2)

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    record()
    listener.stop()
    listener.join()

