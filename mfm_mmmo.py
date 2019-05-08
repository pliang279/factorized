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

import pickle
import time, argparse
import csv
import sys, os
import h5py
from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestClassifier
#from hmmlearn import hmm
from collections import defaultdict, OrderedDict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from keras.utils.np_utils import to_categorical
import json

import sys
from mfm_model import MFM
from mfm_model import MFM_KL, MFM_KL_EF

def get_data(config):
	# labels
	labels_loc = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/annotations/annotations_full.csv'
	labels = dict()
	with open(labels_loc) as csvfile:
		f = csv.reader(open(labels_loc, 'rU'), dialect=csv.excel_tab)
		i = 0
		for line in f:
			line = line[0].split(',')
			i += 1
			if i == 1:
				continue
			link = line[0]
			try:
				label = float(line[-1])
			except:
				label = float(line[1])
			try:
				video_id = link[:link.index('.')]
			except:
				video_id = link
			labels[video_id] = dict()
			labels[video_id]['1'] = label


	# audio_dir = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Raw/Audio/Full/'
	# arr = os.listdir(audio_dir)
	# for file in arr:
	# 	if file.endswith('.wav'):
	# 		covarep_loc = audio_dir + file[:-4] + '.mat'
	# 		if not os.path.exists(covarep_loc):
	# 			print 'cp %s /media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Raw/Audio/temp/'% (audio_dir + file)
	# assert False

	p2fa_dir = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Aligned/'
	arr = os.listdir(p2fa_dir)
	good = 0
	tot = 0
	times = []
	for file in arr:
		video_id = file[:-5]
		segment_id = '1'
		p2fa_loc = p2fa_dir + file
		with open(p2fa_loc) as f:
			ff = f.read().splitlines()
			i = 0
			for line in ff:
				i += 1
				if i == 4:
					start = line
				if i == 5:
					end = line
				if i == 6:
					break
		p2fa_line = "%s,%s,%s,%s,%s"% (video_id,segment_id,start,end,p2fa_loc)
		#print p2fa_line
		covarep_loc = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Features/Audio/covarep/%s.mat' %video_id
		facet_loc = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Features/Video/facet/%s.csv' %video_id
		words_loc = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Features/Text/words/%s_%s.csv' %(video_id,segment_id)
		embeddings_loc = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Features/Text/embeddings/%s_%s.csv' %(video_id,segment_id)
		final_line = "%s,%s,%s,%s,%s,%s,%s,%s"% (video_id,segment_id,start,end,covarep_loc,facet_loc,words_loc,embeddings_loc)
		if os.path.exists(covarep_loc) and os.path.exists(facet_loc) and os.path.exists(words_loc):
			final_line = "%s,%s,%s,%s,%s,%s,%s,%s"% (video_id,segment_id,start,end,covarep_loc,facet_loc,words_loc,embeddings_loc)
			time = float(end)-float(start)
			times.append(time)
			#print final_line
			good += 1
		tot += 1
		#print labels[video_id]['1']
	#print good,tot
	#print sum(times)/float(len(times))
	


	# p2fa_csv_fpath = "MMMO_p2fa.csv"
	# glove_path = '/media/bighdd2/paul/wordemb/scripts/experiments/dan/data/glove.840B.300d.txt'
	# p = P2FA_Helper(p2fa_csv_fpath,output_dir="./", embed_type = "glove", embed_model_path=glove_path)
	# p.load()
	# assert False

	# get sums_dict to segment video into sentence and average across sentence

	sums_dict = dict()
	trans_dir = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Raw/Transcript/Full/'
	arr = os.listdir(trans_dir)
	for file in arr:
		file_dir = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/Main/Raw/Transcript/Full/%s' % file
		with open(file_dir) as trans:
			sens = trans.read().splitlines()
			sums = []
			total = 0
			for sen in sens:
				if len(sen) == 0 or sen[0] == '<' or sen[0] == '/':
					continue
				sen = sen.strip()
				num_words = len(sen.split())
				sums.append(total)
				total += num_words
		video_id = file[:-4]
		if video_id not in sums_dict:
			sums_dict[video_id] = dict()
		sums_dict[video_id]['1'] = sums

	all_csv_fpath = "MMMO_all.csv"

	# Code for loading
	# d = Dataset(all_csv_fpath)
	# features = d.load()

	# # View modalities
	# print d.modalities # Modalities are numbered as modality_0, modality_1, ....

	# # m_0 -> covarep
	# # m_1 -> facet
	# # m_2 -> words
	# # m_3 -> embeddings

	# features = d.align('modality_2')

	# video_dict = OrderedDict()
	# text_dict = OrderedDict()
	# audio_dict = OrderedDict()

	# def agg_sentence(vector,sums):
	# 	res = []
	# 	for i in xrange(len(sums)-1):
	# 		start = sums[i]
	# 		end = sums[i+1]
	# 		curr = np.mean(vector[start:end,:],axis=0)
	# 		if start != end:
	# 			res.append(curr)
	# 	res = np.array(res)
	# 	return res

	# for video_id in features["modality_3"]:
	# 	print video_id
	# 	for segment_id in features["modality_3"][video_id]:
	# 		x = []
	# 		for feat in features["modality_3"][video_id][segment_id]:
	# 			x.append(feat[2])
	# 		#print np.array(x).shape
	# 		x = agg_sentence(np.array(x),sums_dict[video_id][segment_id])
	# 		#print x.shape
	# 		if video_id not in text_dict:
	# 			text_dict[video_id] = OrderedDict()
	# 		text_dict[video_id][segment_id] = x
	# print 'text_dict loaded'
	# for video_id in features["modality_1"]:
	# 	for segment_id in features["modality_1"][video_id]:
	# 		x = []
	# 		for feat in features["modality_1"][video_id][segment_id]:
	# 			x.append(feat[2])
	# 		#print np.array(x).shape
	# 		x = agg_sentence(np.array(x),sums_dict[video_id][segment_id])
	# 		#print x.shape
	# 		if video_id not in video_dict:
	# 			video_dict[video_id] = OrderedDict()
	# 		video_dict[video_id][segment_id] = x
	# print 'video_dict loaded'
	# for video_id in features["modality_0"]:
	# 	for segment_id in features["modality_0"][video_id]:
	# 		x = []
	# 		for feat in features["modality_0"][video_id][segment_id]:
	# 			x.append(feat[2])
	# 		#print np.array(x).shape
	# 		x = agg_sentence(np.array(x),sums_dict[video_id][segment_id])
	# 		#print x.shape
	# 		if video_id not in audio_dict:
	# 			audio_dict[video_id] = OrderedDict()
	# 		audio_dict[video_id][segment_id] = x
	# print 'audio_dict loaded'

	
	# pickle.dump(text_dict, open("text_dict_s.p","wb"))
	# pickle.dump(audio_dict, open("audio_dict_s.p","wb"))
	# pickle.dump(video_dict, open("video_dict_s.p","wb"))
	# assert False

	text_dict = pickle.load(open("/media/bighdd4/Paul/mosi2/experiments/mmmo/text_dict_s.p","rb"))
	audio_dict = pickle.load(open("/media/bighdd4/Paul/mosi2/experiments/mmmo/audio_dict_s.p","rb"))
	video_dict = pickle.load(open("/media/bighdd4/Paul/mosi2/experiments/mmmo/video_dict_s.p","rb"))

	def pad(data,max_segment_len,t):
		curr = []
		try:
			dim = data.shape[1]
		except:
			if t == 1: 
				return np.zeros((max_segment_len,300))
			if t == 2: 
				return np.zeros((max_segment_len,74))
			if t == 3: 
				return np.zeros((max_segment_len,36))
		if max_segment_len >= len(data):
			for i in xrange(max_segment_len-len(data)):
				curr.append([0 for i in xrange(dim)])
			for vec in data:
				curr.append(vec)
		else:	# max_segment_len < len(text), take last max_segment_len of text
			for vec in data[len(data)-max_segment_len:]:
				curr.append(vec)
		curr = np.array(curr)
		return curr

	all_ids = [video_id for video_id in text_dict]
	#print all_ids
	#assert False
	train_i = [(video_id,segment_id) for video_id in all_ids[:220] for segment_id in text_dict[video_id]]
	valid_i = [(video_id,segment_id) for video_id in all_ids[220:260] for segment_id in text_dict[video_id]]
	test_i = [(video_id,segment_id) for video_id in all_ids[260:] for segment_id in text_dict[video_id]]
	print len(all_ids)
	print len(train_i)
	print len(valid_i)
	print len(test_i)

	# test human prediction error:
	# labels_loc = '/media/bighdd4/Prateek/datasets/aligned_dataset/MMMO/annotations/annotations_full.csv'
	# labels = dict()
	# real = []
	# other = []
	# with open(labels_loc) as csvfile:
	# 	f = csv.reader(open(labels_loc, 'rU'), dialect=csv.excel_tab)
	# 	i = 0
	# 	for line in f:
	# 		line = line[0].split(',')
	# 		i += 1
	# 		if i == 1:
	# 			continue
	# 		link = line[0]
	# 		try:
	# 			video_id = link[:link.index('.')]
	# 		except:
	# 			video_id = link
	# 		if (video_id,'1') in test_i:
	# 			real_l = line[1]
	# 			real.append(float(real_l))
	# 			try:
	# 				other_l = float(line[2])
	# 			except:
	# 				other_l = float(line[1])
	# 			other.append(float(other_l))

	# print len(other)

	# predictions = np.array(other).reshape((len(real),))
	# y_test = np.array(real).reshape((len(real),))
	# mae = np.mean(np.absolute(predictions-y_test))
	# print "mae: ", mae
	# print "corr: ", round(np.corrcoef(predictions,y_test)[0][1],5)
	# print 'mult_acc: ', round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
	# true_label = (y_test > 3.5)
	# predicted_label = (predictions > 3.5)
	# print "Confusion Matrix :"
	# print confusion_matrix(true_label, predicted_label)
	# print "Classification Report :"
	# print classification_report(true_label, predicted_label, digits=5)
	# print "Accuracy ", accuracy_score(true_label, predicted_label)

	# assert False

	max_segment_len = config['seqlength']

	text_train_emb = np.nan_to_num(np.array([pad(text_dict[video_id][segment_id],max_segment_len,1) for (video_id,segment_id) in train_i]))
	covarep_train = np.nan_to_num(np.array([pad(audio_dict[video_id][segment_id],max_segment_len,2) for (video_id,segment_id) in train_i]))
	facet_train = np.nan_to_num(np.array([pad(video_dict[video_id][segment_id],max_segment_len,3) for (video_id,segment_id) in train_i]))
	y_train = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id,segment_id) in train_i]))

	text_valid_emb = np.nan_to_num(np.array([pad(text_dict[video_id][segment_id],max_segment_len,1) for (video_id,segment_id) in valid_i]))
	covarep_valid = np.nan_to_num(np.array([pad(audio_dict[video_id][segment_id],max_segment_len,2) for (video_id,segment_id) in valid_i]))
	facet_valid = np.nan_to_num(np.array([pad(video_dict[video_id][segment_id],max_segment_len,3) for (video_id,segment_id) in valid_i]))
	y_valid = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id,segment_id) in valid_i]))

	text_test_emb = np.nan_to_num(np.array([pad(text_dict[video_id][segment_id],max_segment_len,1) for (video_id,segment_id) in test_i]))
	covarep_test = np.nan_to_num(np.array([pad(audio_dict[video_id][segment_id],max_segment_len,2) for (video_id,segment_id) in test_i]))
	facet_test = np.nan_to_num(np.array([pad(video_dict[video_id][segment_id],max_segment_len,3) for (video_id,segment_id) in test_i]))
	y_test = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id,segment_id) in test_i]))

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
	X_test = np.concatenate((text_test_emb, covarep_test, facet_test), axis=2)	# [300+43+43]

	# print np.min(X_train)
	# print np.max(X_train)
	# print np.min(X_valid)
	# print np.max(X_valid)
	# print np.min(X_test)
	# print np.max(X_test)
	# assert False
	X_train[X_train > 255] = 255.0
	X_train[X_train < -255] = -255.0
	X_valid[X_valid > 255] = 255.0
	X_valid[X_valid < -255] = -255.0
	X_test[X_test > 255] = 255.0
	X_test[X_test < -255] = -255.0
	# print np.min(X_train)
	# print np.max(X_train)
	# print np.min(X_valid)
	# print np.max(X_valid)
	# print np.min(X_test)
	# print np.max(X_test)

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

	model = MFM_KL_EF(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

	optimizer = optim.Adam(model.parameters())
	#optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

	# optimizer = optim.SGD([
	#                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
	#                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
	#             ], momentum=0.9)

	l1_loss = nn.L1Loss()
	l2_loss = nn.MSELoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	l1_loss = l1_loss.to(device)
	l2_loss = l2_loss.to(device)
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	
	def train(model, batchsize, X_train, y_train, optimizer, stage):
		epoch_loss = 0
		model.train()
		total_n = X_train.shape[1]
		num_batches = total_n / batchsize
		for batch in xrange(num_batches):
			start = batch*batchsize
			end = (batch+1)*batchsize
			optimizer.zero_grad()
			batch_X = torch.Tensor(X_train[:,start:end]).cuda()
			batch_y = torch.Tensor(y_train[start:end]).cuda()
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
			disc_loss = l1_loss(y_hat, batch_y)
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
			batch_y = torch.Tensor(y_valid).cuda()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			y_hat = y_hat.squeeze(1)
			epoch_loss = l1_loss(y_hat, batch_y).item()
		return epoch_loss

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
				[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
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
			torch.save(model, 'res_mfm_mmmo/mfn_%d.pt' %rand)
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
			torch.save(model, 'res_mfm_mmmo/mfn_%d.pt' %rand)
		else:
			print epoch, train_loss, valid_loss

	model = torch.load('res_mfm_mmmo/mfn_%d.pt' %rand)

	def score(predictions,y_test):
		mae = np.mean(np.absolute(predictions-y_test))
		print "mae: ", mae
		corr = np.corrcoef(predictions,y_test)[0][1]
		print "corr: ", corr
		mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
		print "mult_acc: ", mult
		f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
		print "mult f_score: ", f_score
		true_label = (y_test > 3.5)
		predicted_label = (predictions > 3.5)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

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

	optimizer = optim.Adam(model.parameters())
	#optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

	# optimizer = optim.SGD([
	#                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
	#                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
	#             ], momentum=0.9)

	l1_loss = nn.L1Loss()
	l2_loss = nn.MSELoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	l1_loss = l1_loss.to(device)
	l2_loss = l2_loss.to(device)
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	
	def train(model, batchsize, X_train, y_train, optimizer):
		epoch_loss = 0
		model.train()
		total_n = X_train.shape[1]
		num_batches = total_n / batchsize
		for batch in xrange(num_batches):
			start = batch*batchsize
			end = (batch+1)*batchsize
			optimizer.zero_grad()
			batch_X = torch.Tensor(X_train[:,start:end]).cuda()
			batch_y = torch.Tensor(y_train[start:end]).cuda()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded

			#print x_l_hat[0,0,:10]
			#print x_l_hat[5,0,:10]
			#print x_l_hat[10,0,:10]
			#print x_l_hat[15,0,:10]
			#assert False
			y_hat = y_hat.squeeze(1)
			mmd_loss = config['lda_mmd']*mmd_loss
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			gen_loss = config['lda_xl']*l2_loss(x_l_hat,x_l)+config['lda_xa']*l2_loss(x_a_hat,x_a)+config['lda_xv']*l2_loss(x_v_hat,x_v)
			disc_loss = l1_loss(y_hat, batch_y)
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
			batch_y = torch.Tensor(y_valid).cuda()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			else:
				decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			y_hat = y_hat.squeeze(1)
			epoch_loss = l1_loss(y_hat, batch_y).item()
		return epoch_loss

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			if config['missing']:
				decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
				[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
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
			torch.save(model, 'res_mfm2/mfn_%d.pt' %rand)
		else:
			print epoch, train_loss, valid_loss

	model = torch.load('res_mfm2/mfn_%d.pt' %rand)

	def score(predictions,y_test):
		mae = np.mean(np.absolute(predictions-y_test))
		print "mae: ", mae
		corr = np.corrcoef(predictions,y_test)[0][1]
		print "corr: ", corr
		mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
		print "mult_acc: ", mult
		f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
		print "mult f_score: ", f_score
		true_label = (y_test > 3.5)
		predicted_label = (predictions > 3.5)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	y_hat = predict(model, X_test)
	print 'scoring y_hat'
	score(y_hat,y_test)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', default='configs/mmmo.json', type=str)
parser.add_argument('--type', default='gd', type=str)	# d, gd, m1, m3
parser.add_argument('--fusion', default='mfn', type=str)	# ef, tf, mv, marn, mfn
args = parser.parse_args()
config = json.load(open(args.config), object_pairs_hook=OrderedDict)

X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(config)

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
	config['lda_xl'] = random.choice([1.0,2.0,5.0,10.0]) #random.choice([0.01,0.1,0.5,1.0])
	config['lda_xa'] = random.choice([1.0,2.0,5.0,10.0]) #random.choice([0.01,0.1,0.5,1.0])
	config['lda_xv'] = random.choice([1.0,2.0,5.0,10.0]) #random.choice([0.01,0.1,0.5,1.0])

	config['type'] = 'kl' #m_a,m_b,m_c,m_d,mfm,s2s,kl
	config['missing'] = 0
	config['output_dim'] = 1
	config["windowsize"] = 2
	config["batchsize"] = random.choice([32,64,128])
	config["num_epochs"] = 50
	config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01,0.02])
	config["momentum"] = 0.9 #random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
	NN1Config = dict()
	NN1Config["shapes"] = 128 #random.choice([32,64,128,256])
	NN1Config["drop"] = 0.5 #random.choice([0.0,0.2,0.5,0.7])
	NN2Config = dict()
	NN2Config["shapes"] = 128 #random.choice([32,64,128,256])
	NN2Config["drop"] = 0.5 #random.choice([0.0,0.2,0.5,0.7])
	gamma1Config = dict()
	gamma1Config["shapes"] = 128 #random.choice([32,64,128,256])
	gamma1Config["drop"] = 0.5 #random.choice([0.0,0.2,0.5,0.7])
	gamma2Config = dict()
	gamma2Config["shapes"] = 128 #random.choice([32,64,128,256])
	gamma2Config["drop"] = 0.5 #random.choice([0.0,0.2,0.5,0.7])
	outConfig = dict()
	outConfig["shapes"] = 64 #random.choice([32,64,128,256])
	outConfig["drop"] = 0.5 #random.choice([0.0,0.2,0.5,0.7])
	configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
	print configs
	train_beta_vae(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	#train_mfm(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
