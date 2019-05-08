import numpy as np
seed = 123
np.random.seed(seed)
import random
import torch
torch.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import csv
import os
import sys, h5py
import cPickle, time, argparse, pickle
from sklearn.svm import NuSVC
import random
from sklearn.ensemble import RandomForestClassifier
#from hmmlearn import hmm

from collections import defaultdict, OrderedDict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import json

import sys
from mfm_model import MFM
from mfm_model import MFM_KL, MFM_KL_EF

def get_data(config):
	max_segment_len = config['seqlength']

	s_wv_path = '/media/bighdd4/Prateek/datasets/aligned_dataset/MOUD/SBW-vectors-300-min5.txt'

	# get labels
	labels_file = "/media/bighdd4/Prateek/datasets/aligned_dataset/MOUD/MOUD/Labels/cats.txt"
	labels = dict()
	with open(labels_file) as f_labels:
		for line in f_labels:
			line = line.split()
			name = line[0]
			i = name[name.index('_')+1:].index('_') + name.index('_')+1
			#print i
			video_id = name[:i]
			segment_id = name[i+1:]
			#print video_id, segment_id
			#assert False
			l = line[1]
			if l == 'positive':
				lab = 1
			elif l == 'neutral':	# ignore neutral labels
				continue
			elif l == 'negative':
				lab = 0
			if video_id not in labels:
				labels[video_id] = dict()
			labels[video_id][segment_id] = lab

	# times = []
	# with open("/home/pliangtemp/temp/MOUD_all.csv") as csvfile:
	# 	reader = csv.reader(csvfile)
	# 	for row in reader:
	# 		try:
	# 			times.append(float(row[3])-float(row[2]))
	# 		except:
	# 			pass
	# print sum(times)/float(len(times))
	# assert False

	# split test, train
	all_index_path = '/media/bighdd4/Prateek/datasets/aligned_dataset/MOUD/spanish_text.csv'
	train_index_path = '/media/bighdd4/Prateek/datasets/aligned_dataset/MOUD/train_index.csv'
	test_index_path ='/media/bighdd4/Prateek/datasets/aligned_dataset/MOUD/test_index.csv'
	int2vidseg = dict()
	train_i = []
	valid_i = []
	test_i = []
	videos = set()
	with open(all_index_path) as csvfile:
		reader = csv.reader(csvfile)
		done_videos = []
		for row in reader:
			video_id = row[0][:row[0].index('.')]
			videos.add(video_id)
			segment_id = row[0][row[0].index('.')+5:]
			if len(videos) >= 59:	# 79 videos 49 train 10 val 20 test
				try:
					a = labels[video_id][segment_id]	# neutral ones
					test_i.append((video_id, segment_id))
				except:
					pass
			elif len(videos) >= 49:
				try:
					a = labels[video_id][segment_id]	# neutral ones
					valid_i.append((video_id, segment_id))
				except:
					pass
			else:
				try:
					a = labels[video_id][segment_id]	# neutral ones
					train_i.append((video_id, segment_id))
				except:
					pass
	# train_i = []
	# with open(train_index_path) as train_f:
	# 	for line in train_f:
	# 		try:
	# 			(video_id, segment_id) = int2vidseg[int(line)]
	# 			a = labels[video_id][segment_id]	# neutral ones
	# 			train_i.append(int2vidseg[int(line)])
	# 		except:
	# 			print 'error on line', int(line)
	# test_i = []
	# with open(test_index_path) as test_f:
	# 	for line in test_f:
	# 		try:
	# 			(video_id, segment_id) = int2vidseg[int(line)]
	# 			a = labels[video_id][segment_id]	# neutral ones
	# 			test_i.append(int2vidseg[int(line)])
	# 		except:
	# 			print 'error on line', int(line)
	#print train_i
	#print test_i

	#pickle.dump(labels, open("labels.p","wb"))

	#assert False

	# Arguments for Dataset class
	# csv_fpath = "/home/pliangtemp/temp/MOUD_all.csv"
	# # Code for loading
	# d = Dataset(csv_fpath)
	# features = d.load()

	# # View modalities
	# print d.modalities # Modalities are numbered as modality_0, modality_1, ....

	# # View features for a particular segment of a modality
	# modality = "modality_2" # replace 0 with 1, 2, .... for different modalities, modality_3 is word level alignment. covarep,facet,embeddings,words
	
	# features = d.align('modality_3')

	# video_dict = dict()
	# text_dict = dict()
	# audio_dict = dict()

	# for video_id in features["modality_2"]:
	# 	for segment_id in features["modality_2"][video_id]:
	# 		x = []
	# 		for feat in features["modality_2"][video_id][segment_id]:
	# 			x.append(feat[2])
	# 		x = np.array(x)
	# 		if video_id not in text_dict:
	# 			text_dict[video_id] = dict()
	# 		text_dict[video_id][segment_id] = x
	# print 'text_dict loaded'
	# for video_id in features["modality_1"]:
	# 	for segment_id in features["modality_1"][video_id]:
	# 		x = []
	# 		for feat in features["modality_1"][video_id][segment_id]:
	# 			x.append(feat[2])
	# 		x = np.array(x)
	# 		if video_id not in video_dict:
	# 			video_dict[video_id] = dict()
	# 		video_dict[video_id][segment_id] = x
	# print 'video_dict loaded'
	# for video_id in features["modality_0"]:
	# 	for segment_id in features["modality_0"][video_id]:
	# 		x = []
	# 		for feat in features["modality_0"][video_id][segment_id]:
	# 			x.append(feat[2])
	# 		x = np.array(x)
	# 		if video_id not in audio_dict:
	# 			audio_dict[video_id] = dict()
	# 		audio_dict[video_id][segment_id] = x
	# print 'audio_dict loaded'

	# # m_0 -> covarep
	# # m_1 -> facet
	# # m_2 -> embeddings
	# # m_3 -> words

	# pickle.dump(text_dict, open("text_dict.p","wb"))
	# pickle.dump(audio_dict, open("audio_dict.p","wb"))
	# pickle.dump(video_dict, open("video_dict.p","wb"))

	#assert False

	text_dict = pickle.load(open("/media/bighdd4/Paul/mosi2/experiments/moud/text_dict.p","rb"))
	audio_dict = pickle.load(open("/media/bighdd4/Paul/mosi2/experiments/moud/audio_dict.p","rb"))
	video_dict = pickle.load(open("/media/bighdd4/Paul/mosi2/experiments/moud/video_dict.p","rb"))

	def pad(data,max_segment_len):
		curr = []
		dim = data.shape[1]
		if max_segment_len >= len(data):
			for vec in data:
				curr.append(vec)
			for i in xrange(max_segment_len-len(data)):
				curr.append([0 for i in xrange(dim)])
		else:	# max_segment_len < len(text), take last max_segment_len of text
			for vec in data[len(data)-max_segment_len:]:
				curr.append(vec)
		curr = np.array(curr)
		return curr

	# lengths_train = np.array([len(text_dict[video_id][segment_id]) for (video_id,segment_id) in train_i])
	# lengths_valid = np.array([len(text_dict[video_id][segment_id]) for (video_id,segment_id) in valid_i])
	# lengths_test = np.array([len(text_dict[video_id][segment_id]) for (video_id,segment_id) in test_i])

	# print lengths_test
	# f = h5py.File("out/moud_lengths_test.hdf5", "w")
	# f.create_dataset('d1',data=lengths_test)
	# f.close()
	# assert False

	text_train_emb = np.array([pad(text_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in train_i])
	covarep_train = np.array([pad(audio_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in train_i])
	facet_train = np.array([pad(video_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in train_i])
	y_train = np.array([labels[video_id][segment_id] for (video_id,segment_id) in train_i])

	text_valid_emb = np.array([pad(text_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in valid_i])
	covarep_valid = np.array([pad(audio_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in valid_i])
	facet_valid = np.array([pad(video_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in valid_i])
	y_valid = np.array([labels[video_id][segment_id] for (video_id,segment_id) in valid_i])

	text_test_emb = np.array([pad(text_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in test_i])
	covarep_test = np.array([pad(audio_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in test_i])
	facet_test = np.array([pad(video_dict[video_id][segment_id],max_segment_len) for (video_id,segment_id) in test_i])
	y_test = np.array([labels[video_id][segment_id] for (video_id,segment_id) in test_i])

	# facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
	# facet_train_max[facet_train_max==0] = 1
	# covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
	# covarep_train_max[covarep_train_max==0] = 1

	# facet_train = facet_train / facet_train_max
	# covarep_train = covarep_train / covarep_train_max
	# facet_valid = facet_valid / facet_train_max
	# covarep_valid = covarep_valid / covarep_train_max
	# facet_test = facet_test / facet_train_max
	# covarep_test = covarep_test / covarep_train_max

	print text_train_emb.shape		# n x seq x 300
	print covarep_train.shape       # n x seq x 74
	print facet_train.shape         # n x seq x 35
	X_train = np.concatenate((text_train_emb, covarep_train, facet_train), axis=2)

	print text_valid_emb.shape		# n x seq x 300
	print covarep_valid.shape       # n x seq x 74
	print facet_valid.shape         # n x seq x 35
	X_valid = np.concatenate((text_valid_emb, covarep_valid, facet_valid), axis=2)

	print text_test_emb.shape      # n x seq x 300
	print covarep_test.shape       # n x seq x 74
	print facet_test.shape         # n x seq x 35
	X_test = np.concatenate((text_test_emb, covarep_test, facet_test), axis=2)	# [300+74+35]
	num_classes = 2 # 0, 1
	y_train = to_categorical(y_train, num_classes)
	y_valid = to_categorical(y_valid, num_classes)
	y_test = to_categorical(y_test, num_classes)

	X_train[X_train > 255] = 255.0
	X_train[X_train < -255] = -255.0
	X_valid[X_valid > 255] = 255.0
	X_valid[X_valid < -255] = -255.0
	X_test[X_test > 255] = 255.0
	X_test[X_test < -255] = -255.0
	return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_beta_vae(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
	[d_l,d_a,d_v] = config['input_dims']
	#if config['missing']:
	#	model = MFM_missing(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	#else:
	#	model = MFM(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

	model = MFM_KL_EF(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

	optimizer = optim.Adam(model.parameters(),lr=config["lr"])
	#optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

	# optimizer = optim.SGD([
	#                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
	#                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
	#             ], momentum=0.9)

	cr_loss = nn.CrossEntropyLoss()
	l2_loss = nn.MSELoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	cr_loss = cr_loss.to(device)
	l2_loss = l2_loss.to(device)
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	
	def train(model, batchsize, X_train, y_train, optimizer, stage):
		epoch_loss = 0
		model.train()
		total_n = X_train.shape[1]
		num_batches = total_n / batchsize
		for batch in xrange(num_batches+1):
			start = batch*batchsize
			try:
				end = (batch+1)*batchsize
			except:
				end = total_n
			optimizer.zero_grad()
			batch_X = torch.Tensor(X_train[:,start:end]).cuda()
			batch_y = torch.Tensor(y_train[start:end]).cuda().long()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			y_hat = y_hat.squeeze(1)
			mmd_loss = config['lda_mmd']*mmd_loss
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			gen_loss = config['lda_xl']*l2_loss(x_l_hat,x_l)+config['lda_xa']*l2_loss(x_a_hat,x_a)+config['lda_xv']*l2_loss(x_v_hat,x_v)
			disc_loss = cr_loss(y_hat, batch_y)
			if stage == 1:
				loss = gen_loss + mmd_loss
			if stage == 2:
				loss = disc_loss + mmd_loss
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		return epoch_loss / num_batches

	def evaluate(model, X_valid, y_valid):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(y_valid).cuda().long()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			y_hat = y_hat.squeeze(1)
			epoch_loss = cr_loss(y_hat, batch_y).item()
			#epoch_acc = (batch_y.eq(torch.argmax(y_hat,dim=1).long())).sum().item()
		return epoch_loss

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
				[x_l_hat_nol,x_a_hat_nol,x_v_hat_nol,y_hat_nol] = decoded_nol
				[x_l_hat_noa,x_a_hat_noa,x_v_hat_noa,y_hat_noa] = decoded_noa
				[x_l_hat_nov,x_a_hat_nov,x_v_hat_nov,y_hat_nov] = decoded_nov
				y_hat_nol = y_hat_nol.squeeze(1).cpu().data.numpy()
				y_hat_noa = y_hat_noa.squeeze(1).cpu().data.numpy()
				y_hat_nov = y_hat_nov.squeeze(1).cpu().data.numpy()
				x_l = batch_X[:,:,:d_l]
				x_a = batch_X[:,:,d_l:d_l+d_a]
				x_v = batch_X[:,:,d_l+d_a:]
				x_l_loss = F.mse_loss(x_l_hat,x_l).item()
				x_a_loss = F.mse_loss(x_a_hat,x_a).item()
				x_v_loss = F.mse_loss(x_v_hat,x_v).item()
				x_l_nol_loss = F.mse_loss(x_l_hat_nol,x_l).item()
				x_a_noa_loss = F.mse_loss(x_a_hat_noa,x_a).item()
				x_v_nov_loss = F.mse_loss(x_v_hat_nov,x_v).item()
				print x_l_loss,x_a_loss,x_v_loss
				print x_l_nol_loss,x_a_noa_loss,x_v_nov_loss
				[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
				y_hat = y_hat.squeeze(1).cpu().data.numpy()
				return y_hat,y_hat_nol,y_hat_noa,y_hat_nov
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
				[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
				y_hat = y_hat.squeeze(1).cpu().data.numpy()
				return y_hat

	best_valid = 999999.0
	rand = random.randint(0,100000)
	for epoch in range(config["num_epochs"]):
		train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, 1)
		valid_loss = evaluate(model, X_valid, y_valid)
		scheduler.step(valid_loss)
		if True: #valid_loss <= best_valid:
			# save model
			best_valid = valid_loss
			print epoch, train_loss, valid_loss, 'saving model'
			torch.save(model, 'res_mfm_moud/mfn_%d.pt' %rand)
		else:
			print epoch, train_loss, valid_loss
	best_valid = 999999.0
	for epoch in range(config["num_epochs"]):
		train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, 2)
		valid_loss = evaluate(model, X_valid, y_valid)
		scheduler.step(valid_loss)
		if True: #valid_loss <= best_valid:
			# save model
			best_valid = valid_loss
			print epoch, train_loss, valid_loss, 'saving model'
			torch.save(model, 'res_mfm_moud/mfn_%d.pt' %rand)
		else:
			print epoch, train_loss, valid_loss

	model = torch.load('res_mfm_moud/mfn_%d.pt' %rand)

	def score(predictions,y_test):
		true_label = y_test
		predicted_label = np.argmax(predictions, axis=1)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	if config['missing']:
		y_hat,y_hat_nol,y_hat_noa,y_hat_nov = predict(model, X_test)
		print 'scoring y_hat_nol'
		score(y_hat_nol,y_test)
		print 'scoring y_hat_noa'
		score(y_hat_noa,y_test)
		print 'scoring y_hat_nov'
		score(y_hat_nov,y_test)
	else:
		y_hat = predict(model, X_test)
	print 'scoring y_hat'
	score(y_hat,y_test)

def train_mfm(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
	[d_l,d_a,d_v] = config['input_dims']
	#if config['missing']:
	#	model = MFM_missing(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	#else:
	#	model = MFM(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

	if config['type'] == 'kl':
		model = MFM_KL_EF(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	else:
		model = MFM(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)


	optimizer = optim.Adam(model.parameters(),lr=config["lr"])
	#optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

	# optimizer = optim.SGD([
	#                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
	#                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
	#             ], momentum=0.9)

	cr_loss = nn.CrossEntropyLoss()
	l2_loss = nn.MSELoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	cr_loss = cr_loss.to(device)
	l2_loss = l2_loss.to(device)
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	
	def train(model, batchsize, X_train, y_train, optimizer):
		epoch_loss = 0
		model.train()
		total_n = X_train.shape[1]
		num_batches = total_n / batchsize
		for batch in xrange(num_batches+1):
			start = batch*batchsize
			try:
				end = (batch+1)*batchsize
			except:
				end = total_n
			optimizer.zero_grad()
			batch_X = torch.Tensor(X_train[:,start:end]).cuda()
			batch_y = torch.Tensor(y_train[start:end]).cuda().long()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			y_hat = y_hat.squeeze(1)
			mmd_loss = config['lda_mmd']*mmd_loss
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			gen_loss = config['lda_xl']*l2_loss(x_l_hat,x_l)+config['lda_xa']*l2_loss(x_a_hat,x_a)+config['lda_xv']*l2_loss(x_v_hat,x_v)
			disc_loss = cr_loss(y_hat, batch_y)
			loss = disc_loss + gen_loss + mmd_loss + missing_loss
			loss.backward()
			optimizer.step()
			epoch_loss += disc_loss.item()
		return epoch_loss / num_batches

	def evaluate(model, X_valid, y_valid):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(y_valid).cuda().long()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			y_hat = y_hat.squeeze(1)
			epoch_loss = cr_loss(y_hat, batch_y).item()
			#epoch_acc = (batch_y.eq(torch.argmax(y_hat,dim=1).long())).sum().item()
		return epoch_loss

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
				[x_l_hat_nol,x_a_hat_nol,x_v_hat_nol,y_hat_nol] = decoded_nol
				[x_l_hat_noa,x_a_hat_noa,x_v_hat_noa,y_hat_noa] = decoded_noa
				[x_l_hat_nov,x_a_hat_nov,x_v_hat_nov,y_hat_nov] = decoded_nov
				y_hat_nol = y_hat_nol.squeeze(1).cpu().data.numpy()
				y_hat_noa = y_hat_noa.squeeze(1).cpu().data.numpy()
				y_hat_nov = y_hat_nov.squeeze(1).cpu().data.numpy()
				x_l = batch_X[:,:,:d_l]
				x_a = batch_X[:,:,d_l:d_l+d_a]
				x_v = batch_X[:,:,d_l+d_a:]
				x_l_loss = F.mse_loss(x_l_hat,x_l).item()
				x_a_loss = F.mse_loss(x_a_hat,x_a).item()
				x_v_loss = F.mse_loss(x_v_hat,x_v).item()
				x_l_nol_loss = F.mse_loss(x_l_hat_nol,x_l).item()
				x_a_noa_loss = F.mse_loss(x_a_hat_noa,x_a).item()
				x_v_nov_loss = F.mse_loss(x_v_hat_nov,x_v).item()
				print x_l_loss,x_a_loss,x_v_loss
				print x_l_nol_loss,x_a_noa_loss,x_v_nov_loss
				[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
				y_hat = y_hat.squeeze(1).cpu().data.numpy()
				return y_hat,y_hat_nol,y_hat_noa,y_hat_nov
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
				[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
				y_hat = y_hat.squeeze(1).cpu().data.numpy()
				return y_hat

	best_valid = 999999.0
	rand = random.randint(0,100000)
	for epoch in range(config["num_epochs"]):
		train_loss = train(model, config["batchsize"], X_train, y_train, optimizer)
		valid_loss = evaluate(model, X_valid, y_valid)
		scheduler.step(valid_loss)
		if valid_loss <= best_valid:
			# save model
			best_valid = valid_loss
			print epoch, train_loss, valid_loss, 'saving model'
			torch.save(model, 'res_mfm_moud/mfn_%d.pt' %rand)
		else:
			print epoch, train_loss, valid_loss

	model = torch.load('res_mfm_moud/mfn_%d.pt' %rand)

	def score(predictions,y_test):
		true_label = y_test
		predicted_label = np.argmax(predictions, axis=1)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	if config['missing']:
		y_hat,y_hat_nol,y_hat_noa,y_hat_nov = predict(model, X_test)
		print 'scoring y_hat_nol'
		score(y_hat_nol,y_test)
		print 'scoring y_hat_noa'
		score(y_hat_noa,y_test)
		print 'scoring y_hat_nov'
		score(y_hat_nov,y_test)
	else:
		y_hat = predict(model, X_test)
	print 'scoring y_hat'
	score(y_hat,y_test)
	
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', default='configs/moud.json', type=str)
parser.add_argument('--type', default='gd', type=str)	# d, gd, m1, m3
parser.add_argument('--fusion', default='mfn', type=str)	# ef, tf, mv, marn, mfn
args = parser.parse_args()
config = json.load(open(args.config), object_pairs_hook=OrderedDict)

X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(config)

y_train = np.argmax(y_train,axis=1)
y_valid = np.argmax(y_valid,axis=1)
y_test = np.argmax(y_test,axis=1)

while True:
	config = dict()
	config["input_dims"] = [300,74,36]
	hl = random.choice([32,64,88,128,156,256])
	ha = random.choice([8,16,32,48,64,80])
	hv = random.choice([8,16,32,48,64,80])
	config["h_dims"] = [hl,ha,hv]
	config['zy_size'] = random.choice([8,16,32,48,64,80])
	config['zl_size'] = random.choice([32,64,88,128,156,256])
	config['za_size'] = random.choice([8,16,32,48,64,80])
	config['zv_size'] = random.choice([8,16,32,48,64,80])
	config['fy_size'] = random.choice([8,16,32,48,64,80])
	config['fl_size'] = random.choice([32,64,88,128,156,256])
	config['fa_size'] = random.choice([8,16,32,48,64,80])
	config['fv_size'] = random.choice([8,16,32,48,64,80])
	config["memsize"] = random.choice([64,128,256,300,400])
	config['zy_to_fy_dropout'] = random.choice([0.0,0.2,0.5,0.7])
	config['zl_to_fl_dropout'] = random.choice([0.0,0.2,0.5,0.7])
	config['za_to_fa_dropout'] = random.choice([0.0,0.2,0.5,0.7])
	config['zv_to_fv_dropout'] = random.choice([0.0,0.2,0.5,0.7])
	config['fy_to_y_dropout'] = random.choice([0.0,0.2,0.5,0.7])

	config['lda_mmd'] = random.choice([10,50,100,200]) #random.choice([0.001,0.005,0.01,0.1,0.5,1.0])
	config['lda_xl'] = random.choice([0.01,0.1,0.5,1.0,5.0])
	config['lda_xa'] = random.choice([0.01,0.1,0.5,1.0,5.0])
	config['lda_xv'] = random.choice([0.01,0.1,0.5,1.0,5.0])

	config['type'] = 'kl'
	config['missing'] = 0
	config['output_dim'] = 2
	config["windowsize"] = 2
	config["batchsize"] = random.choice([32,64,128])
	config["num_epochs"] = 50
	config["lr"] = random.choice([0.001,0.002,0.004,0.005,0.008,0.01,0.02])
	config["momentum"] = 0.5 #random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
	NN1Config = dict()
	NN1Config["shapes"] = random.choice([32,64,128])
	NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	NN2Config = dict()
	NN2Config["shapes"] = random.choice([32,64,128])
	NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	gamma1Config = dict()
	gamma1Config["shapes"] = random.choice([32,64,128])
	gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	gamma2Config = dict()
	gamma2Config["shapes"] = random.choice([32,64,128])
	gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	outConfig = dict()
	outConfig["shapes"] = random.choice([32,64,128])
	outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])

	configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
	print configs
	train_beta_vae(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	#train_mfm(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
