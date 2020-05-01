#! /usr/bin/env python3
import numpy as np
import pickle 
import rospkg
import rospy
from rospy.numpy_msg import numpy_msg
import socket 
import sys
import time
import threading
from celex5_msgs.msg import event, eventData, eventVector

import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from os.path import dirname
import torch

DEBUG = 8

from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy
if DEBUG>0:
    from utils.models1 import Classifier
else:
    from utils.models import Classifier
from utils.dataset import NCaltech101

# Define global vairables
fig = plt.figure()
ax = plt.axes(xlim=(0, 224), ylim=(0, 224))
datasetClasses = None
pred_label = torch.tensor([0])
img_frame = np.zeros((224,224))
img_lock = threading.Lock()
event_data = None
msg_lock = threading.Lock()
PATH = ""

def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1280)

    parser.add_argument("--checkpoint", default="log/final/model_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--test_dataset", default="N-Caltech101/testing/")

    flags = parser.parse_args()

    checkpoint_loc = PATH + "/" + flags.checkpoint
    dataset_loc = PATH + "/" + flags.test_dataset

    assert os.path.isdir(dirname(checkpoint_loc)), f"Checkpoint {checkpoint_loc} not found."

    print(f"----------------------------\n"
            f"Starting testing with \n"
            f"height: {flags.height}\n"
            f"width: {flags.width}\n"
            f"checkpoint: {checkpoint_loc}\n"
            f"device: {flags.device}\n"
            f"test_dataset: {dataset_loc}\n"
            f"----------------------------")

    return flags

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def updateImg(i):
    ax.clear()
    img_lock.acquire()
	img_label = pred_label[0].item()
	img_to_show = img_frame.copy()
	img_lock.release()
	ax.set_title( datasetClasses[img_label])
    img = ax.imshow(img_to_show, cmap='gray')
    return [img]

@threaded
def get_events_and_predict():
    global img_frame, pred_label, datasetClasses
    events = []
    while not rospy.is_shutdown():
        if event_data == None:
            continue
        msg_lock.acquire()
        npX = np.fromiter(event_data.x, dtype=np.float32) 
        npY = np.fromiter(event_data.y, dtype=np.float32)
        msg_lock.release()
        npT = np.zeros_like(npY)
        npP = np.zeros_like(npY)
        ev = np.stack([npX, npY, npT, npP], axis=-1)
        ev = np.concatenate([ev, np.zeros((len(ev),1), dtype=np.float32)],1)
        ev = np.expand_dims(ev, axis=0)
        events.append(ev)
        with torch.no_grad():
            pred, img = model(events)
        img_lock.acquire()
        img_frame = img.squeeze().cpu().numpy()
        pred_label = pred.argmax(1)
        img_lock.release()
        print(datasetClasses[ pred_label[0].item()])
        events.clear()

def eventVector_cb(msg):
    global event_data
    msg_lock.acquire()
    event_data = msg
    msg_lock.release()
    print(len(event_data.x))

if __name__ == '__main__':
    rospy.init_node("events2npy")
    rospack = rospkg.RosPack()
    PATH = rospack.get_path('celex5_monocular')
    rospy.loginfo("NODE: events2npy starts , using Python %s" % sys.version)

    events_sub = rospy.Subscriber("/celex_monocular/celex5_event", numpy_msg(eventData), eventVector_cb, queue_size=1)

    flags = FLAGS()
    dataset_loc = PATH + "/" + flags.test_dataset
    checkpoint_loc = PATH + "/" + flags.checkpoint
    dim = (flags.height, flags.width)

    # Obtain classes from test dataset
    test_dataset = NCaltech101(dataset_loc, resolution=dim)
    datasetClasses = test_dataset.getClasses()

    # model, load and put to device
    model = Classifier(device=flags.device, dimension=dim)
    model = model.to(flags.device)
    ckpt = torch.load(checkpoint_loc)
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval()
    model.setMode(1)

    anim = FuncAnimation(fig, updateImg, frames=2000, interval=100)
    events_handler = get_events_and_predict()

    plt.show()

    events_handler.join()

    while not rospy.is_shutdown():
        pass
