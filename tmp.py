from pynput import keyboard
import threading

# Initialize the one hot set with no keys pressed
one_hot_set = set()

# Define a dictionary mapping each key to its index in the one hot set
key_index_map = {'Key.up': 0, 'Key.down': 1, 'Key.left': 2, 'Key.right': 3, "'c'": 4, "'x'": 5}

# Define a lock to synchronize access to the one hot set
lock = threading.Lock()

def update_one_hot_set(key, value):
    with lock:
        # If the pressed/released key is one of the keys we're interested in, update the one hot set
        if key in key_index_map:
            if value:
                one_hot_set.add(key_index_map[key])
            else:
                one_hot_set.discard(key_index_map[key])

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

def print_one_hot_vector():
    with lock:
        print(sorted(list(one_hot_set)))
    threading.Timer(1, print_one_hot_vector).start()

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print_one_hot_vector()
    listener.join()
