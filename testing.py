import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import dirname
import random
import threading 
import torch
import torchvision
import tqdm
from utils.loss import cross_entropy_loss_and_accuracy

# Define constants
DEBUG = 8

if DEBUG>0:
    from utils.models1 import Classifier
else:
    from utils.models import Classifier
from utils.dataset import NCaltech101
from utils.loader import Loader
from utils.train_eval import train_one_epoch, eval_one_epoch

# Define global vairables
running = True
fig = plt.figure()
ax = plt.axes(xlim=(0, 224), ylim=(0, 224))
datasetClasses = None
events = []
pred_label = torch.tensor([0])
pred_correct = False
img_frame = np.zeros((224,224))
img_lock = threading.Lock()

if DEBUG>=8:
    if DEBUG==9:
        seed = 1586354275
    else:
        import time
        seed = int( time.time())
    print("Seed: %d" % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=240)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint", default="log/final/model_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test_dataset", default="N-Caltech101/testing")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.test_dataset} not found."

    print(f"----------------------------\n"
            f"Starting testing with \n"
            f"height: {flags.height}\n"
            f"width: {flags.width}\n"
            f"batch_size: {flags.batch_size}\n"
            f"checkpoint: {flags.checkpoint}\n"
            f"device: {flags.device}\n"
            f"test_dataset: {flags.test_dataset}\n"
            f"----------------------------")

    return flags

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def sigint_handler(signal, frame):
	global running
	running = False

def updateImg(i):
    ax.clear()
    if pred_correct:
	    ax.set_title( datasetClasses[ pred_label[0].item()], 
                        backgroundcolor='green',
                        color='white',
                        fontsize=20)
    else:
        ax.set_title( datasetClasses[ pred_label[0].item()], 
                        backgroundcolor='red',
                        color='white',
                        fontsize=20)
    img_lock.acquire()
    img = ax.imshow(img_frame, cmap='gray')
    img_lock.release()
    return [img]

@threaded
def get_events_and_predict(model, device, dataloader):
    global running, img_frame, pred_label, pred_correct
    sum_accuracy = 0
    sum_loss = 0
    
    for events, labels in tqdm.tqdm(dataloader):
        if running == False:
            break

        labels = labels.to(device)
        
        with torch.no_grad():
            pred, representation = model(events)
            loss, accuracy = cross_entropy_loss_and_accuracy(pred, labels)

        img_lock.acquire()
        img_frame = representation.squeeze().cpu().numpy()
        pred_label = pred.argmax(1)
        pred_correct = (pred_label == labels)[0]
        img_lock.release()

        sum_accuracy += accuracy
        sum_loss += loss

    validation_loss = sum_loss.item() / len(dataloader)
    validation_accuracy = sum_accuracy.item() / len(dataloader)
    print(f"Test Loss: {validation_loss}, Test Accuracy: {validation_accuracy}")

if __name__ == '__main__':
    flags = FLAGS()
    dim = (flags.height, flags.width)

    # datasets
    test_dataset = NCaltech101(flags.test_dataset)
    datasetClasses = test_dataset.getClasses()

    # construct loader, responsible for streaming data to gpu
    test_loader = Loader(test_dataset, flags, flags.device)

    # model, load and put to device
    model = Classifier(device=flags.device, dimension=dim)
    ckpt = torch.load(flags.checkpoint, map_location=flags.device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(flags.device)

    model = model.eval()
    model.setMode(1)

    anim = FuncAnimation(fig, updateImg, frames=2000, interval=10)
    events_prediction = get_events_and_predict(model, flags.device, test_loader)
    
    plt.show()

    events_prediction.join()
