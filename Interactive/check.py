import pickle
import torch

with open('./test.pickle', 'rb') as f:
    tmp = pickle.load(f)
    print(tmp)
    for i in tmp:
        print(i)
    