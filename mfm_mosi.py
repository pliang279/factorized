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

import data_loader as loader
from collections import defaultdict, OrderedDict
import argparse
import cPickle as pickle
import time
import json, os, ast, h5py

from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import sys
from mfm_model import M_A, M_B, M_C, M_D, MFM, MFM_missing, seq2seq, basic_missing
from mfm_model import MFM_KL, MFM_KL_EF

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', default='configs/mosi.json', type=str)
parser.add_argument('--type', default='mgddm', type=str)    # d, gd, m1, m3
parser.add_argument('--fusion', default='mfn', type=str)    # ef, tf, mv, marn, mfn
parser.add_argument('-s', '--feature_selection', default=1, type=int, choices=[0,1], help='whether to use feature_selection')
args = parser.parse_args()
config = json.load(open(args.config), object_pairs_hook=OrderedDict)

def get_data(args,config):
	tr_split = 2.0/3                        # fixed. 62 training & validation, 31 test
	val_split = 0.1514                      # fixed. 52 training 10 validation
	use_pretrained_word_embedding = True    # fixed. use glove 300d
	embedding_vecor_length = 300            # fixed. use glove 300d
	# 115                   # fixed for MOSI. The max length of a segment in MOSI dataset is 114 
	max_segment_len = config['seqlength']
	end_to_end = True                       # fixed

	word2ix = loader.load_word2ix()
	word_embedding = [loader.load_word_embedding()] if use_pretrained_word_embedding else None
	train, valid, test = loader.load_word_level_features(max_segment_len, tr_split)

	ix2word = inv_map = {v: k for k, v in word2ix.iteritems()}
	print len(word2ix)
	print len(ix2word)
	print word_embedding[0].shape

	feature_str = ''
	if args.feature_selection:
		with open('/media/bighdd5/Paul/mosi/fs_mask.pkl') as f:
			[covarep_ix, facet_ix] = pickle.load(f)
		facet_train = train['facet'][:,:,facet_ix]
		facet_valid = valid['facet'][:,:,facet_ix]
		facet_test = test['facet'][:,:,facet_ix]
		covarep_train = train['covarep'][:,:,covarep_ix]
		covarep_valid = valid['covarep'][:,:,covarep_ix]
		covarep_test = test['covarep'][:,:,covarep_ix]
		feature_str = '_t'+str(embedding_vecor_length) + '_c'+str(covarep_test.shape[2]) + '_f'+str(facet_test.shape[2])
	else:
		facet_train = train['facet']
		facet_valid = valid['facet']
		covarep_train = train['covarep'][:,:,1:35]
		covarep_valid = valid['covarep'][:,:,1:35]
		facet_test = test['facet']
		covarep_test = test['covarep'][:,:,1:35]

	text_train = train['text']
	text_valid = valid['text']
	text_test = test['text']
	y_train = train['label']
	y_valid = valid['label']
	y_test = test['label']

	lengths_train = train['lengths']
	lengths_valid = valid['lengths']
	lengths_test = test['lengths']

	#f = h5py.File("out/mosi_lengths_test.hdf5", "w")
	#f.create_dataset('d1',data=lengths_test)
	#f.close()
	#assert False

	facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
	facet_train_max[facet_train_max==0] = 1
	#covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
	#covarep_train_max[covarep_train_max==0] = 1

	facet_train = facet_train / facet_train_max
	facet_valid = facet_valid / facet_train_max
	#covarep_train = covarep_train / covarep_train_max
	facet_test = facet_test / facet_train_max
	#covarep_test = covarep_test / covarep_train_max

	text_input = Input(shape=(max_segment_len,), dtype='int32', name='text_input')
	text_eb_layer = Embedding(word_embedding[0].shape[0], embedding_vecor_length, input_length=max_segment_len, weights=word_embedding, name = 'text_eb_layer', trainable=False)(text_input)
	model = Model(text_input, text_eb_layer)
	text_train_emb = model.predict(text_train)
	print text_train_emb.shape      # n x seq x 300
	print covarep_train.shape       # n x seq x 5/34
	print facet_train.shape         # n x seq x 20/43
	X_train = np.concatenate((text_train_emb, covarep_train, facet_train), axis=2)

	text_valid_emb = model.predict(text_valid)
	print text_valid_emb.shape      # n x seq x 300
	print covarep_valid.shape       # n x seq x 5/34
	print facet_valid.shape         # n x seq x 20/43
	X_valid = np.concatenate((text_valid_emb, covarep_valid, facet_valid), axis=2)

	text_test_emb = model.predict(text_test)
	print text_test_emb.shape      # n x seq x 300
	print covarep_test.shape       # n x seq x 5/34
	print facet_test.shape         # n x seq x 20/43
	X_test = np.concatenate((text_test_emb, covarep_test, facet_test), axis=2)

	return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_data_missing(args,config):
	tr_split = 2.0/3                        # fixed. 62 training & validation, 31 test
	val_split = 0.1514                      # fixed. 52 training 10 validation
	use_pretrained_word_embedding = True    # fixed. use glove 300d
	embedding_vecor_length = 300            # fixed. use glove 300d
	# 115                   # fixed for MOSI. The max length of a segment in MOSI dataset is 114 
	max_segment_len = config['seqlength']
	end_to_end = True                       # fixed

	word2ix = loader.load_word2ix()
	word_embedding = [loader.load_word_embedding()] if use_pretrained_word_embedding else None
	train, valid, test = loader.load_word_level_features(max_segment_len, tr_split)

	ix2word = inv_map = {v: k for k, v in word2ix.iteritems()}
	print len(word2ix)
	print len(ix2word)
	print word_embedding[0].shape

	feature_str = ''
	if args.feature_selection:
		with open('/media/bighdd5/Paul/mosi/fs_mask.pkl') as f:
			[covarep_ix, facet_ix] = pickle.load(f)
		facet_train = train['facet'][:,:,facet_ix]
		facet_valid = valid['facet'][:,:,facet_ix]
		facet_test = test['facet'][:,:,facet_ix]
		covarep_train = train['covarep'][:,:,covarep_ix]
		covarep_valid = valid['covarep'][:,:,covarep_ix]
		covarep_test = test['covarep'][:,:,covarep_ix]
		feature_str = '_t'+str(embedding_vecor_length) + '_c'+str(covarep_test.shape[2]) + '_f'+str(facet_test.shape[2])
	else:
		facet_train = train['facet']
		facet_valid = valid['facet']
		covarep_train = train['covarep'][:,:,1:35]
		covarep_valid = valid['covarep'][:,:,1:35]
		facet_test = test['facet']
		covarep_test = test['covarep'][:,:,1:35]

	text_train = train['text']
	text_valid = valid['text']
	text_test = test['text']
	y_train = train['label']
	y_valid = valid['label']
	y_test = test['label']

	lengths_train = train['lengths']
	lengths_valid = valid['lengths']
	lengths_test = test['lengths']

	#f = h5py.File("out/mosi_lengths_test.hdf5", "w")
	#f.create_dataset('d1',data=lengths_test)
	#f.close()
	#assert False

	facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
	facet_train_max[facet_train_max==0] = 1
	covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
	covarep_train_max[covarep_train_max==0] = 1

	facet_train = facet_train / facet_train_max
	covarep_train = covarep_train / covarep_train_max
	facet_valid = facet_valid / facet_train_max
	covarep_valid = covarep_valid / covarep_train_max
	facet_test = facet_test / facet_train_max
	covarep_test = covarep_test / covarep_train_max

	text_input = Input(shape=(max_segment_len,), dtype='int32', name='text_input')
	text_eb_layer = Embedding(word_embedding[0].shape[0], embedding_vecor_length, input_length=max_segment_len, weights=word_embedding, name = 'text_eb_layer', trainable=False)(text_input)
	model = Model(text_input, text_eb_layer)
	text_train_emb = model.predict(text_train)
	print text_train_emb.shape      # n x seq x 300
	print covarep_train.shape       # n x seq x 5/34
	print facet_train.shape         # n x seq x 20/43

	#print np.min(facet_train)
	#print np.max(facet_train)
	#print np.min(covarep_train)
	#print np.max(covarep_train)
	#print np.min(text_train_emb)
	#print np.max(text_train_emb)
	#assert False

	X_train = np.concatenate((text_train_emb, covarep_train, facet_train), axis=2)

	text_valid_emb = model.predict(text_valid)
	print text_valid_emb.shape      # n x seq x 300
	print covarep_valid.shape       # n x seq x 5/34
	print facet_valid.shape         # n x seq x 20/43
	X_valid = np.concatenate((text_valid_emb, covarep_valid, facet_valid), axis=2)

	text_test_emb = model.predict(text_test)
	print text_test_emb.shape      # n x seq x 300
	print covarep_test.shape       # n x seq x 5/34
	print facet_test.shape         # n x seq x 20/43
	X_test = np.concatenate((text_test_emb, covarep_test, facet_test), axis=2)

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
		true_label = (y_test >= 0)
		predicted_label = (predictions >= 0)
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
	
	if config['type'] == 'kl':
		model = MFM_KL(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
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
			decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
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
		true_label = (y_test >= 0)
		predicted_label = (predictions >= 0)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	y_hat = predict(model, X_test)
	print 'scoring y_hat'
	score(y_hat,y_test)

def train_mfm_test_zeros(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
	[d_l,d_a,d_v] = config['input_dims']
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
			decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
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
			batch_X_nol = torch.cat([torch.zeros_like(batch_X[:,:,:d_l]),batch_X[:,:,d_l:]],dim=2)
			batch_X_noa = torch.cat([batch_X[:,:,:d_l],torch.zeros_like(batch_X[:,:,d_l:d_l+d_a]),batch_X[:,:,d_l+d_a:]],dim=2)
			batch_X_nov = torch.cat([batch_X[:,:,:d_l+d_a],torch.zeros_like(batch_X[:,:,d_l+d_a:])],dim=2)
			decoded_nol,mmd_loss,missing_loss = model.forward(batch_X_nol)
			decoded_noa,mmd_loss,missing_loss = model.forward(batch_X_noa)
			decoded_nov,mmd_loss,missing_loss = model.forward(batch_X_nov)
			[x_l_hat_nol,x_a_hat_nol,x_v_hat_nol,y_hat_nol] = decoded_nol
			[x_l_hat_noa,x_a_hat_noa,x_v_hat_noa,y_hat_noa] = decoded_noa
			[x_l_hat_nov,x_a_hat_nov,x_v_hat_nov,y_hat_nov] = decoded_nov
			y_hat_nol = y_hat_nol.squeeze(1).cpu().data.numpy()
			y_hat_noa = y_hat_noa.squeeze(1).cpu().data.numpy()
			y_hat_nov = y_hat_nov.squeeze(1).cpu().data.numpy()
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			x_l_nol_loss = F.mse_loss(x_l_hat_nol,x_l).item()
			x_a_noa_loss = F.mse_loss(x_a_hat_noa,x_a).item()
			x_v_nov_loss = F.mse_loss(x_v_hat_nov,x_v).item()
			print x_l_nol_loss,x_a_noa_loss,x_v_nov_loss
			return y_hat_nol,y_hat_noa,y_hat_nov

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
		true_label = (y_test >= 0)
		predicted_label = (predictions >= 0)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	y_hat_nol,y_hat_noa,y_hat_nov = predict(model, X_test)
	print 'scoring y_hat_nol'
	score(y_hat_nol,y_test)
	print 'scoring y_hat_noa'
	score(y_hat_noa,y_test)
	print 'scoring y_hat_nov'
	score(y_hat_nov,y_test)

def train_mfm_ablation(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
	[d_l,d_a,d_v] = config['input_dims']
	if config['type'] == 'm_a':
		model = M_A(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	if config['type'] == 'm_b':
		model = M_B(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	if config['type'] == 'm_c':
		model = M_C(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	if config['type'] == 'm_d':
		model = M_D(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

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
			decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
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
			decoded,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			x_l_loss = F.mse_loss(x_l_hat,x_l).item()
			x_a_loss = F.mse_loss(x_a_hat,x_a).item()
			x_v_loss = F.mse_loss(x_v_hat,x_v).item()
			print x_l_loss,x_a_loss,x_v_loss
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
		true_label = (y_test >= 0)
		predicted_label = (predictions >= 0)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	y_hat = predict(model, X_test)
	print 'scoring y_hat'
	score(y_hat,y_test)

def train_seq2seq(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
	[d_l,d_a,d_v] = config['input_dims']
	model = seq2seq(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	
	optimizer = optim.Adam(model.parameters())
	#optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

	# optimizer = optim.SGD([
	#                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
	#                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
	#             ], momentum=0.9)

	l2_loss = nn.MSELoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
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
			decoded_nol,decoded_noa,decoded_nov,mmd_loss = model.forward(batch_X)
			[x_l_hat_nol] = decoded_nol
			[x_a_hat_noa] = decoded_noa
			[x_v_hat_nov] = decoded_nov
			#print x_l_hat[0,0,:10]
			#print x_l_hat[5,0,:10]
			#print x_l_hat[10,0,:10]
			#print x_l_hat[15,0,:10]
			#assert False
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			gen_loss = config['lda_xl']*l2_loss(x_l_hat_nol,x_l) \
					  +config['lda_xa']*l2_loss(x_a_hat_noa,x_a) \
					  +config['lda_xv']*l2_loss(x_v_hat_nov,x_v) \
					  +config['lda_mmd']*mmd_loss
			loss = gen_loss
			loss.backward()
			optimizer.step()
			epoch_loss += gen_loss.item()
		return epoch_loss / num_batches

	def evaluate(model, X_valid, y_valid):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(y_valid).cuda()
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			decoded_nol,decoded_noa,decoded_nov,mmd_loss = model.forward(batch_X)
			[x_l_hat_nol] = decoded_nol
			[x_a_hat_noa] = decoded_noa
			[x_v_hat_nov] = decoded_nov
			epoch_loss = config['lda_xl']*l2_loss(x_l_hat_nol,x_l) \
					    +config['lda_xa']*l2_loss(x_a_hat_noa,x_a) \
					    +config['lda_xv']*l2_loss(x_v_hat_nov,x_v)
		return epoch_loss.item()

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			decoded_nol,decoded_noa,decoded_nov,mmd_loss = model.forward(batch_X)
			[x_l_hat_nol] = decoded_nol
			[x_a_hat_noa] = decoded_noa
			[x_v_hat_nov] = decoded_nov
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			x_l_nol_loss = F.mse_loss(x_l_hat_nol,x_l).item()
			x_a_noa_loss = F.mse_loss(x_a_hat_noa,x_a).item()
			x_v_nov_loss = F.mse_loss(x_v_hat_nov,x_v).item()
			#print x_l_loss,x_a_loss,x_v_loss
			print x_l_nol_loss,x_a_noa_loss,x_v_nov_loss
			return
			
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
		true_label = (y_test >= 0)
		predicted_label = (predictions >= 0)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	if config['type'] == 's2s':
		predict(model, X_test)
		return

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
	return
	
def train_mfm_missing(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
	[d_l,d_a,d_v] = config['input_dims']
	model = MFM_missing(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	
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
			decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			[x_l_hat_nol,x_a_hat_nol,x_v_hat_nol,y_hat_nol] = decoded_nol
			[x_l_hat_noa,x_a_hat_noa,x_v_hat_noa,y_hat_noa] = decoded_noa
			[x_l_hat_nov,x_a_hat_nov,x_v_hat_nov,y_hat_nov] = decoded_nov
			y_hat = y_hat.squeeze(1)
			y_hat_nol = y_hat_nol.squeeze(1)
			y_hat_noa = y_hat_noa.squeeze(1)
			y_hat_nov = y_hat_nov.squeeze(1)
			mmd_loss = config['lda_mmd']*mmd_loss
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			gen_loss = config['lda_xl']*l2_loss(x_l_hat,x_l) \
			          +config['lda_xa']*l2_loss(x_a_hat,x_a) \
			          +config['lda_xv']*l2_loss(x_v_hat,x_v) \
			          +config['lda_xl']*l2_loss(x_l_hat_nol,x_l) \
			          +config['lda_xa']*l2_loss(x_a_hat_noa,x_a) \
			          +config['lda_xv']*l2_loss(x_v_hat_noa,x_v)
			disc_loss = l1_loss(y_hat, batch_y) \
			           +l1_loss(y_hat_nol, batch_y) \
			           +l1_loss(y_hat_noa, batch_y) \
			           +l1_loss(y_hat_nov, batch_y)
			loss = disc_loss + gen_loss + mmd_loss + missing_loss
			loss.backward()
			optimizer.step()
			epoch_loss += l2_loss(x_l_hat,x_l).item()
		return epoch_loss / num_batches

	def evaluate(model, X_valid, y_valid):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(y_valid).cuda()
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss = model.forward(batch_X)
			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			[x_l_hat_nol,x_a_hat_nol,x_v_hat_nol,y_hat_nol] = decoded_nol
			[x_l_hat_noa,x_a_hat_noa,x_v_hat_noa,y_hat_noa] = decoded_noa
			[x_l_hat_nov,x_a_hat_nov,x_v_hat_nov,y_hat_nov] = decoded_nov
			y_hat = y_hat.squeeze(1)
			y_hat_nol = y_hat_nol.squeeze(1)
			y_hat_noa = y_hat_noa.squeeze(1)
			y_hat_nov = y_hat_nov.squeeze(1)
			mmd_loss = config['lda_mmd']*mmd_loss
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			gen_loss = config['lda_xl']*l2_loss(x_l_hat,x_l) \
			          +config['lda_xa']*l2_loss(x_a_hat,x_a) \
			          +config['lda_xv']*l2_loss(x_v_hat,x_v) \
			          +config['lda_xl']*l2_loss(x_l_hat_nol,x_l) \
			          +config['lda_xa']*l2_loss(x_a_hat_noa,x_a) \
			          +config['lda_xv']*l2_loss(x_v_hat_noa,x_v)
			disc_loss = l1_loss(y_hat, batch_y) \
			           +l1_loss(y_hat_nol, batch_y) \
			           +l1_loss(y_hat_noa, batch_y) \
			           +l1_loss(y_hat_nov, batch_y)
			loss = disc_loss + gen_loss + mmd_loss + missing_loss
			epoch_loss = loss.item()
		return epoch_loss

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
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
			print 'all present',x_l_loss,x_a_loss,x_v_loss

			x_l_nol_loss = F.mse_loss(x_l_hat_nol,x_l).item()
			x_a_nol_loss = F.mse_loss(x_a_hat_nol,x_a).item()
			x_v_nol_loss = F.mse_loss(x_v_hat_nol,x_v).item()
			print 'l missing',x_l_nol_loss,x_a_nol_loss,x_v_nol_loss

			x_l_noa_loss = F.mse_loss(x_l_hat_noa,x_l).item()
			x_a_noa_loss = F.mse_loss(x_a_hat_noa,x_a).item()
			x_v_noa_loss = F.mse_loss(x_v_hat_noa,x_v).item()
			print 'a missing',x_l_noa_loss,x_a_noa_loss,x_v_noa_loss
			
			x_l_nov_loss = F.mse_loss(x_l_hat_nov,x_l).item()
			x_a_nov_loss = F.mse_loss(x_a_hat_nov,x_a).item()
			x_v_nov_loss = F.mse_loss(x_v_hat_nov,x_v).item()
			print 'v missing',x_l_nov_loss,x_a_nov_loss,x_v_nov_loss

			[x_l_hat,x_a_hat,x_v_hat,y_hat] = decoded
			y_hat = y_hat.squeeze(1).cpu().data.numpy()
			return y_hat,y_hat_nol,y_hat_noa,y_hat_nov
		
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
		true_label = (y_test >= 0)
		predicted_label = (predictions >= 0)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	y_hat,y_hat_nol,y_hat_noa,y_hat_nov = predict(model, X_test)
	print 'scoring y_hat_nol'
	score(y_hat_nol,y_test)
	print 'scoring y_hat_noa'
	score(y_hat_noa,y_test)
	print 'scoring y_hat_nov'
	score(y_hat_nov,y_test)	
	print 'scoring y_hat'
	score(y_hat,y_test)

def train_basic_missing(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
	[d_l,d_a,d_v] = config['input_dims']
	model = basic_missing(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
	
	optimizer = optim.Adam(model.parameters())
	#optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

	# optimizer = optim.SGD([
	#                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
	#                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
	#             ], momentum=0.9)

	l1_loss = nn.L1Loss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	l1_loss = l1_loss.to(device)
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
			y_hat_nol,y_hat_noa,y_hat_nov,mmd_loss = model.forward(batch_X)
			y_hat_nol = y_hat_nol.squeeze(1)
			y_hat_noa = y_hat_noa.squeeze(1)
			y_hat_nov = y_hat_nov.squeeze(1)
			x_l = batch_X[:,:,:d_l]
			x_a = batch_X[:,:,d_l:d_l+d_a]
			x_v = batch_X[:,:,d_l+d_a:]
			disc_loss = l1_loss(y_hat_nol, batch_y) \
					   +l1_loss(y_hat_noa, batch_y) \
					   +l1_loss(y_hat_nov, batch_y) \
					   +config['lda_mmd']*mmd_loss
			loss = disc_loss
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
			y_hat_nol,y_hat_noa,y_hat_nov,mmd_loss = model.forward(batch_X)
			y_hat_nol = y_hat_nol.squeeze(1)
			epoch_loss = l1_loss(y_hat_nol, batch_y).item()
		return epoch_loss

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			y_hat_nol,y_hat_noa,y_hat_nov,mmd_loss = model.forward(batch_X)
			y_hat_nol = y_hat_nol.squeeze(1).cpu().data.numpy()
			y_hat_noa = y_hat_noa.squeeze(1).cpu().data.numpy()
			y_hat_nov = y_hat_nov.squeeze(1).cpu().data.numpy()
			return y_hat_nol,y_hat_noa,y_hat_nov

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
		true_label = (y_test >= 0)
		predicted_label = (predictions >= 0)
		print "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print "Accuracy ", accuracy_score(true_label, predicted_label)
		sys.stdout.flush()

	y_hat_nol,y_hat_noa,y_hat_nov = predict(model, X_test)
	print 'scoring y_hat_nol'
	score(y_hat_nol,y_test)
	print 'scoring y_hat_noa'
	score(y_hat_noa,y_test)
	print 'scoring y_hat_nov'
	score(y_hat_nov,y_test)

def best_acc(X_train, y_train, X_valid, y_valid, X_test, y_test):
	#[{'batchsize': 32, 'num_epochs': 200, 'zv_to_fv_dropout': 0.7, 
	#'memsize': 64, 'fy_size': 16, 'fa_size': 8, 'lr': 0.01, 
	#'zl_to_fl_dropout': 0.2, 'momentum': 0.9, 'fv_size': 8, 
	#'zy_size': 32, 'input_dims': [300, 5, 20], 'zl_size': 32, 
	#'fy_to_y_dropout': 0.0, 'za_size': 8, 'h_dims': [88, 64, 48], 
	#'za_to_fa_dropout': 0.2, 'lda_xa': 0.01, 'fl_size': 88, 
	#'windowsize': 2, 'lda_xl': 1.0, 'zy_to_fy_dropout': 0.0, 
	#'lda_mmd': 1.0, 'zv_size': 80, 'lda_xv': 0.5}, 
	#{'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.5}, 
	#{'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.5}, 
	#{'shapes': 64, 'drop': 0.5}]
	config = dict()
	config["input_dims"] = [300,5,20]
	hl = 88 #random.choice([32,64,88,128,156,256])
	ha = 64 #random.choice([8,16,32,48,64,80])
	hv = 48 #random.choice([8,16,32,48,64,80])
	config["h_dims"] = [hl,ha,hv]
	config['zy_size'] = 32 #random.choice([8,16,32,48,64,80])
	config['zl_size'] = 32 #random.choice([32,64,88,128,156,256])
	config['za_size'] = 8 #random.choice([8,16,32,48,64,80])
	config['zv_size'] = 80 #random.choice([8,16,32,48,64,80])
	config['fy_size'] = 16 #random.choice([8,16,32,48,64,80])
	config['fl_size'] = 88 #random.choice([32,64,88,128,156,256])
	config['fa_size'] = 8 #random.choice([8,16,32,48,64,80])
	config['fv_size'] = 8 #random.choice([8,16,32,48,64,80])
	config["memsize"] = 64 #random.choice([64,128,256,300,400])
	config['zy_to_fy_dropout'] = 0.0 #random.choice([0.0,0.2,0.5,0.7])
	config['zl_to_fl_dropout'] = 0.2 #random.choice([0.0,0.2,0.5,0.7])
	config['za_to_fa_dropout'] = 0.2 #random.choice([0.0,0.2,0.5,0.7])
	config['zv_to_fv_dropout'] = 0.7 #random.choice([0.0,0.2,0.5,0.7])
	config['fy_to_y_dropout'] = 0.0 #random.choice([0.0,0.2,0.5,0.7])

	config['lda_mmd'] = 1.0 #random.choice([0.01,0.1,0.5,1.0])
	config['lda_xl'] = 1.0 #random.choice([0.01,0.1,0.5,1.0])
	config['lda_xa'] = 0.01 #random.choice([0.01,0.1,0.5,1.0])
	config['lda_xv'] = 0.5 #random.choice([0.01,0.1,0.5,1.0])

	config['missing'] = 0
	config["windowsize"] = 2
	config["batchsize"] = 32 #random.choice([32,64,128])
	config["num_epochs"] = 30
	config["lr"] = 0.01 #random.choice([0.001,0.002,0.005,0.008,0.01,0.02])
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
	train_mfm(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)

#print 'RUNNING MISSING MODALITY EXPERIMENTS'
#X_train, y_train, X_valid, y_valid, X_test, y_test = get_data_missing(args,config)
X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args,config)
sys.stdout.flush()

#print np.mean(np.square(y_test))
#print np.mean(np.square(y_test-np.mean(y_test)))
#assert False

#best_acc(X_train, y_train, X_valid, y_valid, X_test, y_test)
#assert False

while True:
	config = dict()
	config["input_dims"] = [300,5,20]
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

	config['lda_mmd'] = random.choice([10,50,100,200]) #1.0 #random.choice([0.001,0.005,0.01,0.1,0.5,1.0])
	config['lda_xl'] = random.choice([0.01,0.1,0.5,1.0,2.0,5.0,10.0]) #random.choice([0.01,0.1,0.5,1.0])
	config['lda_xa'] = random.choice([0.01,0.1,0.5,1.0,2.0,5.0,10.0]) #random.choice([0.01,0.1,0.5,1.0])
	config['lda_xv'] = random.choice([0.01,0.1,0.5,1.0,2.0,5.0,10.0]) #random.choice([0.01,0.1,0.5,1.0])

	config['type'] = 'kl' #m_a,m_b,m_c,m_d,mfm,s2s,bm,kl
	config['missing'] = 0
	config['zeros'] = 0
	config['output_dim'] = 1
	config["windowsize"] = 2
	config["batchsize"] = random.choice([32,64,128])
	config["num_epochs"] = 50
	config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01,0.02])
	config["momentum"] = 0.9 #random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
	NN1Config = dict()
	NN1Config["shapes"] = random.choice([32,64,128,256])
	NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	NN2Config = dict()
	NN2Config["shapes"] = random.choice([32,64,128,256])
	NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	gamma1Config = dict()
	gamma1Config["shapes"] = random.choice([32,64,128,256])
	gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	gamma2Config = dict()
	gamma2Config["shapes"] = random.choice([32,64,128,256])
	gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	outConfig = dict()
	outConfig["shapes"] = random.choice([32,64,128,256])
	outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
	configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
	print configs
	train_beta_vae(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	continue

	if config['missing'] == 1 and config['type'] == 'bm':
		train_basic_missing(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	elif config['missing'] == 1 and config['type'] == 'mfm':
		train_mfm_missing(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	elif config['missing'] == 1 and config['type'] == 's2s':
		train_seq2seq(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	elif config['zeros'] == 1 and config['type'] == 'mfm':
		train_mfm_test_zeros(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	elif config['type'] == 'mfm' or config['type'] == 'kl':
		train_mfm(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
	else:
		train_mfm_ablation(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)

