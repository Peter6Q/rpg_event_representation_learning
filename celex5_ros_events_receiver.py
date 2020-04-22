#! /usr/bin/env python3
import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import dirname
import signal
import socket
import threading 
import torch

# Define constants
COMM_IP = 'localhost'
COMM_PORT = 8484 
DEBUG = 8

from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy
if DEBUG>0:
    from utils.models1 import Classifier
else:
    from utils.models import Classifier
from utils.dataset import NCaltech101

# Define global vairables
running = True
fig = plt.figure()
ax = plt.axes(xlim=(0, 224), ylim=(0, 224))
datasetClasses = None
events = []
pred_label = torch.tensor([0])
img_frame = np.zeros((224,224))
img_lock = threading.Lock()

def FLAGS():
	parser = argparse.ArgumentParser(
		"""Deep Learning for Events. Supply a config file.""")

	# can be set in config
	parser.add_argument("--height", type=int, default=800)
	parser.add_argument("--width", type=int, default=1280)
	
	parser.add_argument("--checkpoint", default="log/final/model_best.pth")
	parser.add_argument("--data_loc", 
		default="celex5/celex5_ros/src/celex5_monocular/output/eventRecord.npy")
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--pin_memory", type=bool, default=True)
	parser.add_argument("--test_dataset", default="N-Caltech101/testing/")

	flags = parser.parse_args()

	assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."

	print(f"----------------------------\n"
			f"Starting testing with \n"
			f"height: {flags.height}\n"
			f"width: {flags.width}\n"
			f"checkpoint: {flags.checkpoint}\n"
			f"data_loc: {flags.data_loc}\n"
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
	ax.set_title( datasetClasses[ pred_label[0].item()])
	img_lock.acquire()
	img = ax.imshow(img_frame, cmap='gray')
	img_lock.release()
	return [img]

@threaded
def get_events_and_predict():
	global running, img_frame, pred_label, datasetClasses
	comm = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	comm.connect((COMM_IP,COMM_PORT))
	comm.settimeout(5)
	while running:	
		comm.send(b'\x00') 
		response = comm.recv(1)
		if response != b'\x00':
			continue
		ev = np.load(flags.data_loc).astype(np.float32)
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
	comm.send(b'\xff') 
	comm.close()

if __name__ == '__main__': 
	flags = FLAGS()
	dim = (flags.height, flags.width)

	# Obtain classes from test dataset
	test_dataset = NCaltech101(flags.test_dataset, resolution=dim)
	datasetClasses = test_dataset.getClasses()

	# model, load and put to device
	model = Classifier(device=flags.device, dimension=dim)
	model = model.to(flags.device)
	ckpt = torch.load(flags.checkpoint)
	model.load_state_dict(ckpt["state_dict"])
	model = model.eval()
	model.setMode(1)

	anim = FuncAnimation(fig, updateImg, frames=2000, interval=100)
	signal.signal(signal.SIGINT, sigint_handler)
	events_handler = get_events_and_predict()
	
	plt.show()

	events_handler.join()
