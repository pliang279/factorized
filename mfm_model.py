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
import sys

def compute_kernel(x, y):
	x_size = x.size(0)
	y_size = y.size(0)
	dim = x.size(1)
	x = x.unsqueeze(1) # (x_size, 1, dim)
	y = y.unsqueeze(0) # (1, y_size, dim)
	tiled_x = x.expand(x_size, y_size, dim)
	tiled_y = y.expand(x_size, y_size, dim)
	kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
	return torch.exp(-kernel_input) # (x_size, y_size)

def loss_MMD(zy):
	zy_real_gauss = Variable(torch.randn(zy.size())) # no need to be the same size

	#if args.cuda:
	zy_real_gauss = zy_real_gauss.cuda()
	zy_real_kernel = compute_kernel(zy_real_gauss, zy_real_gauss)
	zy_fake_kernel = compute_kernel(zy, zy)
	zy_kernel = compute_kernel(zy_real_gauss, zy)
	zy_mmd = zy_real_kernel.mean() + zy_fake_kernel.mean() - 2.0*zy_kernel.mean()
	return zy_mmd

def loss_KLD(mu, logvar):
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return KLD

class encoderLSTM(nn.Module):
	def __init__(self, d, h): #, n_layers, bidirectional, dropout):
		super(encoderLSTM, self).__init__()
		self.lstm = nn.LSTMCell(d, h)
		self.fc1 = nn.Linear(h, h)
		self.h = h

	def forward(self, x):
		# x is t x n x h
		t = x.shape[0]
		n = x.shape[1]
		self.hx = torch.zeros(n, self.h).cuda()
		self.cx = torch.zeros(n, self.h).cuda()
		all_hs = []
		all_cs = []
		for i in range(t):
			self.hx, self.cx = self.lstm(x[i], (self.hx, self.cx))
			all_hs.append(self.hx)
			all_cs.append(self.cx)
		# last hidden layer last_hs is n x h
		last_hs = all_hs[-1]
		last_hs = self.fc1(last_hs)
		return last_hs

class decoderLSTM(nn.Module):
	def __init__(self, h, d):
		super(decoderLSTM, self).__init__()
		self.lstm = nn.LSTMCell(h, h)
		self.fc1 = nn.Linear(h, d)
		self.d = d
		self.h = h
		
	def forward(self, hT, t): # only embedding vector
		# x is n x d
		n = hT.shape[0]
		h = hT.shape[1]
		self.hx = torch.zeros(n, self.h).cuda()
		self.cx = torch.zeros(n, self.h).cuda()
		final_hs = []
		all_hs = []
		all_cs = []
		for i in range(t):
			if i == 0:
				self.hx, self.cx = self.lstm(hT, (self.hx, self.cx))
			else:
				self.hx, self.cx = self.lstm(all_hs[-1], (self.hx, self.cx))
			all_hs.append(self.hx)
			all_cs.append(self.cx)
			final_hs.append(self.hx.view(1,n,h))
		final_hs = torch.cat(final_hs, dim=0)
		all_recons = self.fc1(final_hs)
		return all_recons

class MFN(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(MFN, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		self.mem_dim = config["memsize"]
		window_dim = config["windowsize"]
		output_dim = config['output_dim']
		attInShape = total_h_dim*window_dim
		gammaInShape = attInShape+self.mem_dim
		final_out = total_h_dim+self.mem_dim
		h_att1 = NN1Config["shapes"]
		h_att2 = NN2Config["shapes"]
		h_gamma1 = gamma1Config["shapes"]
		h_gamma2 = gamma2Config["shapes"]
		h_out = outConfig["shapes"]
		att1_dropout = NN1Config["drop"]
		att2_dropout = NN2Config["drop"]
		gamma1_dropout = gamma1Config["drop"]
		gamma2_dropout = gamma2Config["drop"]
		out_dropout = outConfig["drop"]

		self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
		self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
		self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

		self.att1_fc1 = nn.Linear(attInShape, h_att1)
		self.att1_fc2 = nn.Linear(h_att1, attInShape)
		self.att1_dropout = nn.Dropout(att1_dropout)

		self.att2_fc1 = nn.Linear(attInShape, h_att2)
		self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
		self.att2_dropout = nn.Dropout(att2_dropout)

		self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
		self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
		self.gamma1_dropout = nn.Dropout(gamma1_dropout)

		self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
		self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
		self.gamma2_dropout = nn.Dropout(gamma2_dropout)

		self.out_fc1 = nn.Linear(final_out, h_out)
		self.out_fc2 = nn.Linear(h_out, output_dim)
		self.out_dropout = nn.Dropout(out_dropout)
		
	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]
		self.h_l = torch.zeros(n, self.dh_l).cuda()
		self.h_a = torch.zeros(n, self.dh_a).cuda()
		self.h_v = torch.zeros(n, self.dh_v).cuda()
		self.c_l = torch.zeros(n, self.dh_l).cuda()
		self.c_a = torch.zeros(n, self.dh_a).cuda()
		self.c_v = torch.zeros(n, self.dh_v).cuda()
		self.mem = torch.zeros(n, self.mem_dim).cuda()
		all_h_ls = []
		all_h_as = []
		all_h_vs = []
		all_c_ls = []
		all_c_as = []
		all_c_vs = []
		all_mems = []
		for i in range(t):
			# prev time step
			prev_c_l = self.c_l
			prev_c_a = self.c_a
			prev_c_v = self.c_v
			# curr time step
			new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
			new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
			new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
			# concatenate
			prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
			new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
			cStar = torch.cat([prev_cs,new_cs], dim=1)
			attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
			attended = attention*cStar
			cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
			both = torch.cat([attended,self.mem], dim=1)
			gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
			gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
			self.mem = gamma1*self.mem + gamma2*cHat
			all_mems.append(self.mem)
			# update
			self.h_l, self.c_l = new_h_l, new_c_l
			self.h_a, self.c_a = new_h_a, new_c_a
			self.h_v, self.c_v = new_h_v, new_c_v
			all_h_ls.append(self.h_l)
			all_h_as.append(self.h_a)
			all_h_vs.append(self.h_v)
			all_c_ls.append(self.c_l)
			all_c_as.append(self.c_a)
			all_c_vs.append(self.c_v)

		# last hidden layer last_hs is n x h
		last_h_l = all_h_ls[-1]
		last_h_a = all_h_as[-1]
		last_h_v = all_h_vs[-1]
		last_mem = all_mems[-1]
		last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
		return last_hs

class M_A(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(M_A, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_l = encoderLSTM(self.d_l+self.d_a+self.d_v,zl_size)

		self.decoder_l = decoderLSTM(fy_size+fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fy_size+fl_size,self.d_a)
		self.decoder_v = decoderLSTM(fy_size+fl_size,self.d_v)
		
		self.mfn_encoder = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
		self.last_to_zy_fc1 = nn.Linear(last_mfn_size,zy_size)

		self.zy_to_fy_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_to_fy_fc2 = nn.Linear(fy_size,fy_size)
		self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.fy_to_y_fc1 = nn.Linear(fy_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		zl = self.encoder_l.forward(x)
		mfn_last = self.mfn_encoder.forward(x)
		zy = self.last_to_zy_fc1(mfn_last)
		mmd_loss = loss_MMD(zl)+loss_MMD(zy)
		missing_loss = 0.0

		fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
		fyfl = torch.cat([fy,fl], dim=1)

		dec_len = t
		x_l_hat = self.decoder_l.forward(fyfl,dec_len)
		x_a_hat = self.decoder_a.forward(fyfl,dec_len)
		x_v_hat = self.decoder_v.forward(fyfl,dec_len)
		y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]

		return decoded,mmd_loss,missing_loss

class M_B(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(M_B, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_l = encoderLSTM(self.d_l,zl_size)
		self.encoder_a = encoderLSTM(self.d_a,za_size)
		self.encoder_v = encoderLSTM(self.d_v,zv_size)

		self.decoder_l = decoderLSTM(fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fa_size,self.d_a)
		self.decoder_v = decoderLSTM(fv_size,self.d_v)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

		self.fy_to_y_fc1 = nn.Linear(fl_size+fa_size+fv_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		zl = self.encoder_l.forward(x_l)
		za = self.encoder_a.forward(x_a)
		zv = self.encoder_v.forward(x_v)
		mmd_loss = loss_MMD(zl)+loss_MMD(za)+loss_MMD(zv)
		missing_loss = 0.0

		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
		fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
		fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))
		
		dec_len = t
		x_l_hat = self.decoder_l.forward(fl,dec_len)
		x_a_hat = self.decoder_a.forward(fa,dec_len)
		x_v_hat = self.decoder_v.forward(fv,dec_len)
		fy = torch.cat([fl,fa,fv],dim=1)
		y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]

		return decoded,mmd_loss,missing_loss

class M_C(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(M_C, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.decoder_l = decoderLSTM(fy_size,self.d_l)
		self.decoder_a = decoderLSTM(fy_size,self.d_a)
		self.decoder_v = decoderLSTM(fy_size,self.d_v)
		
		self.mfn_encoder = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
		self.last_to_zy_fc1 = nn.Linear(last_mfn_size,zy_size)

		self.zy_to_fy_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_to_fy_fc2 = nn.Linear(fy_size,fy_size)
		self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

		self.fy_to_y_fc1 = nn.Linear(fy_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		mfn_last = self.mfn_encoder.forward(x)
		zy = self.last_to_zy_fc1(mfn_last)
		mmd_loss = loss_MMD(zy)
		missing_loss = 0.0
		fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))

		dec_len = t
		x_l_hat = self.decoder_l.forward(fy,dec_len)
		x_a_hat = self.decoder_a.forward(fy,dec_len)
		x_v_hat = self.decoder_v.forward(fy,dec_len)
		y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]

		return decoded,mmd_loss,missing_loss

class M_D(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(M_D, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_l = encoderLSTM(self.d_l,zl_size)
		self.encoder_a = encoderLSTM(self.d_a,za_size)
		self.encoder_v = encoderLSTM(self.d_v,zv_size)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

		self.fs_to_y = nn.Linear(fl_size+fa_size+fv_size,output_dim)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		zl = self.encoder_l.forward(x_l)
		za = self.encoder_a.forward(x_a)
		zv = self.encoder_v.forward(x_v)
		mmd_loss = 0.0
		missing_loss = 0.0

		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
		fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
		fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))

		fs = torch.cat([fl,fa,fv],dim=1)
		y_hat = self.fs_to_y(fs)
		decoded = [x_l,x_a,x_v,y_hat]

		return decoded,mmd_loss,missing_loss

class MFM(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(MFM, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_l = encoderLSTM(self.d_l,zl_size)
		self.encoder_a = encoderLSTM(self.d_a,za_size)
		self.encoder_v = encoderLSTM(self.d_v,zv_size)

		self.decoder_l = decoderLSTM(fy_size+fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fy_size+fa_size,self.d_a)
		self.decoder_v = decoderLSTM(fy_size+fv_size,self.d_v)
		
		self.mfn_encoder = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
		self.last_to_zy_fc1 = nn.Linear(last_mfn_size,zy_size)

		self.zy_to_fy_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_to_fy_fc2 = nn.Linear(fy_size,fy_size)
		self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

		self.fy_to_y_fc1 = nn.Linear(fy_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		zl = self.encoder_l.forward(x_l)
		za = self.encoder_a.forward(x_a)
		zv = self.encoder_v.forward(x_v)

		mfn_last = self.mfn_encoder.forward(x)
		zy = self.last_to_zy_fc1(mfn_last)
		mmd_loss = loss_MMD(zl)+loss_MMD(za)+loss_MMD(zv)+loss_MMD(zy)
		missing_loss = 0.0

		fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
		fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
		fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))
		
		fyfl = torch.cat([fy,fl], dim=1)
		fyfa = torch.cat([fy,fa], dim=1)
		fyfv = torch.cat([fy,fv], dim=1)

		dec_len = t
		x_l_hat = self.decoder_l.forward(fyfl,dec_len)
		x_a_hat = self.decoder_a.forward(fyfa,dec_len)
		x_v_hat = self.decoder_v.forward(fyfv,dec_len)
		y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]

		return decoded,mmd_loss,missing_loss

class MFM_KL_EF(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(MFM_KL_EF, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_l = encoderLSTM(self.d_l,zl_size)
		self.encoder_a = encoderLSTM(self.d_a,za_size)
		self.encoder_v = encoderLSTM(self.d_v,zv_size)

		self.decoder_l = decoderLSTM(fy_size+fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fy_size+fa_size,self.d_a)
		self.decoder_v = decoderLSTM(fy_size+fv_size,self.d_v)

		last_ef_size = zl_size+za_size+zv_size
		self.ef_encoder = encoderLSTM(self.d_l+self.d_a+self.d_v,last_ef_size)
		self.last_to_zy_fc1 = nn.Linear(last_ef_size,zy_size)
		self.last_to_logvarzy_fc1 = nn.Linear(last_ef_size,zy_size)

		self.last_to_zl_fc1 = nn.Linear(zl_size,zl_size)
		self.last_to_za_fc1 = nn.Linear(za_size,za_size)
		self.last_to_zv_fc1 = nn.Linear(zv_size,zv_size)
		self.last_to_logvarzl_fc1 = nn.Linear(zl_size,zl_size)
		self.last_to_logvarza_fc1 = nn.Linear(za_size,za_size)
		self.last_to_logvarzv_fc1 = nn.Linear(zv_size,zv_size)

		self.zy_to_fy_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_to_fy_fc2 = nn.Linear(fy_size,fy_size)
		self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

		self.fy_to_y_fc1 = nn.Linear(fy_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		zl_last = self.encoder_l.forward(x_l)
		za_last = self.encoder_a.forward(x_a)
		zv_last = self.encoder_v.forward(x_v)
		zl = self.last_to_zl_fc1(zl_last)
		za = self.last_to_za_fc1(za_last)
		zv = self.last_to_zv_fc1(zv_last)
		logvar_zl = self.last_to_logvarzl_fc1(zl_last)
		logvar_za = self.last_to_logvarza_fc1(za_last)
		logvar_zv = self.last_to_logvarzv_fc1(zv_last)

		ef_last = self.ef_encoder.forward(x)
		zy = self.last_to_zy_fc1(ef_last)
		logvar_zy = self.last_to_logvarzy_fc1(ef_last)

		kld_loss = loss_KLD(zl,logvar_zl)+loss_KLD(za,logvar_za)+loss_KLD(zv,logvar_zv)+loss_KLD(zy,logvar_zy)
		missing_loss = 0.0

		fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
		fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
		fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))
		
		fyfl = torch.cat([fy,fl], dim=1)
		fyfa = torch.cat([fy,fa], dim=1)
		fyfv = torch.cat([fy,fv], dim=1)

		dec_len = t
		x_l_hat = self.decoder_l.forward(fyfl,dec_len)
		x_a_hat = self.decoder_a.forward(fyfa,dec_len)
		x_v_hat = self.decoder_v.forward(fyfv,dec_len)
		y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]

		return decoded,kld_loss,missing_loss

class MFM_KL(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(MFM_KL, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_l = encoderLSTM(self.d_l,zl_size)
		self.encoder_a = encoderLSTM(self.d_a,za_size)
		self.encoder_v = encoderLSTM(self.d_v,zv_size)

		self.decoder_l = decoderLSTM(fy_size+fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fy_size+fa_size,self.d_a)
		self.decoder_v = decoderLSTM(fy_size+fv_size,self.d_v)
		
		self.mfn_encoder = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
		self.last_to_zy_fc1 = nn.Linear(last_mfn_size,zy_size)
		self.last_to_logvarzy_fc1 = nn.Linear(last_mfn_size,zy_size)

		self.last_to_zl_fc1 = nn.Linear(zl_size,zl_size)
		self.last_to_za_fc1 = nn.Linear(za_size,za_size)
		self.last_to_zv_fc1 = nn.Linear(zv_size,zv_size)
		self.last_to_logvarzl_fc1 = nn.Linear(zl_size,zl_size)
		self.last_to_logvarza_fc1 = nn.Linear(za_size,za_size)
		self.last_to_logvarzv_fc1 = nn.Linear(zv_size,zv_size)

		self.zy_to_fy_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_to_fy_fc2 = nn.Linear(fy_size,fy_size)
		self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

		self.fy_to_y_fc1 = nn.Linear(fy_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		zl_last = self.encoder_l.forward(x_l)
		za_last = self.encoder_a.forward(x_a)
		zv_last = self.encoder_v.forward(x_v)
		zl = self.last_to_zl_fc1(zl_last)
		za = self.last_to_za_fc1(za_last)
		zv = self.last_to_zv_fc1(zv_last)
		logvar_zl = self.last_to_logvarzl_fc1(zl_last)
		logvar_za = self.last_to_logvarza_fc1(za_last)
		logvar_zv = self.last_to_logvarzv_fc1(zv_last)

		mfn_last = self.mfn_encoder.forward(x)
		zy = self.last_to_zy_fc1(mfn_last)
		logvar_zy = self.last_to_logvarzy_fc1(mfn_last)

		kld_loss = loss_KLD(zl,logvar_zl)+loss_KLD(za,logvar_za)+loss_KLD(zv,logvar_zv)+loss_KLD(zy,logvar_zy)
		missing_loss = 0.0

		fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
		fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
		fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))
		
		fyfl = torch.cat([fy,fl], dim=1)
		fyfa = torch.cat([fy,fa], dim=1)
		fyfv = torch.cat([fy,fv], dim=1)

		dec_len = t
		x_l_hat = self.decoder_l.forward(fyfl,dec_len)
		x_a_hat = self.decoder_a.forward(fyfa,dec_len)
		x_v_hat = self.decoder_v.forward(fyfv,dec_len)
		y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]

		return decoded,kld_loss,missing_loss

class MFM_missing(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(MFM_missing, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_l = encoderLSTM(self.d_l,zl_size)
		self.encoder_a = encoderLSTM(self.d_a,za_size)
		self.encoder_v = encoderLSTM(self.d_v,zv_size)

		self.encoder_la_to_v = encoderLSTM(self.d_l+self.d_a,zv_size)
		self.encoder_lv_to_a = encoderLSTM(self.d_l+self.d_v,za_size)
		self.encoder_av_to_l = encoderLSTM(self.d_a+self.d_v,zl_size)

		self.encoder_la_to_y = encoderLSTM(self.d_l+self.d_a,zy_size)
		self.encoder_lv_to_y = encoderLSTM(self.d_l+self.d_v,zy_size)
		self.encoder_av_to_y = encoderLSTM(self.d_a+self.d_v,zy_size)

		self.decoder_l = decoderLSTM(fy_size+fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fy_size+fa_size,self.d_a)
		self.decoder_v = decoderLSTM(fy_size+fv_size,self.d_v)
		
		self.mfn_encoder = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)
		self.last_to_zy_fc1 = nn.Linear(last_mfn_size,zy_size)

		self.zy_to_fy_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_to_fy_fc2 = nn.Linear(fy_size,fy_size)
		self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

		self.fy_to_y_fc1 = nn.Linear(fy_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		# zl,za,zv,zy are not activated (mmd loss with gaussian)
		zl = self.encoder_l.forward(x_l)
		za = self.encoder_a.forward(x_a)
		zv = self.encoder_v.forward(x_v)
		mfn_last = self.mfn_encoder.forward(x)
		zy = self.last_to_zy_fc1(mfn_last)

		# zv_nov,za_noa,zl_nol are not activated (mse loss with zl,za,zv)
		zv_nov = self.encoder_la_to_v.forward(torch.cat([x_l,x_a], dim=2))
		za_noa = self.encoder_lv_to_a.forward(torch.cat([x_l,x_v], dim=2))
		zl_nol = self.encoder_av_to_l.forward(torch.cat([x_a,x_v], dim=2))

		# zy_nov,zy_noa,zy_nol are not activated (mse loss with zy)
		zy_nov = self.encoder_la_to_y.forward(torch.cat([x_l,x_a], dim=2))
		zy_noa = self.encoder_lv_to_y.forward(torch.cat([x_l,x_v], dim=2))
		zy_nol = self.encoder_av_to_y.forward(torch.cat([x_a,x_v], dim=2))

		mmd_loss = loss_MMD(zl)+loss_MMD(za)+loss_MMD(zv)+loss_MMD(zy)
		missing_loss = F.mse_loss(zv_nov,zv) \
					 + F.mse_loss(za_noa,za) \
					 + F.mse_loss(zl_nol,zl) \
					 + F.mse_loss(zy_nov,zy) \
					 + F.mse_loss(zy_noa,zy) \
					 + F.mse_loss(zy_nol,zy)

		def decode(zl,za,zv,zy):
			# fl,fa,fv,fy are relu activated
			fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
			fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
			fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
			fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))
			fyfl = torch.cat([fy,fl], dim=1)
			fyfa = torch.cat([fy,fa], dim=1)
			fyfv = torch.cat([fy,fv], dim=1)
			dec_len = t
			x_l_hat = self.decoder_l.forward(fyfl,dec_len)
			x_a_hat = self.decoder_a.forward(fyfa,dec_len)
			x_v_hat = self.decoder_v.forward(fyfv,dec_len)
			y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
			return x_l_hat,x_a_hat,x_v_hat,y_hat

		x_l_hat,x_a_hat,x_v_hat,y_hat = decode(zl,za,zv,zy)
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]
		x_l_hat_nol,x_a_hat_nol,x_v_hat_nol,y_hat_nol = decode(zl_nol,za,zv,zy_nol)
		decoded_nol = [x_l_hat_nol,x_a_hat_nol,x_v_hat_nol,y_hat_nol]
		x_l_hat_noa,x_a_hat_noa,x_v_hat_noa,y_hat_noa = decode(zl,za_noa,zv,zy_noa)
		decoded_noa = [x_l_hat_noa,x_a_hat_noa,x_v_hat_noa,y_hat_noa]
		x_l_hat_nov,x_a_hat_nov,x_v_hat_nov,y_hat_nov = decode(zl,za,zv_nov,zy_nov)
		decoded_nov = [x_l_hat_nov,x_a_hat_nov,x_v_hat_nov,y_hat_nov]

		return decoded,decoded_nol,decoded_noa,decoded_nov,mmd_loss,missing_loss

class seq2seq(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(seq2seq, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_la_to_v = encoderLSTM(self.d_l+self.d_a,zv_size)
		self.encoder_lv_to_a = encoderLSTM(self.d_l+self.d_v,za_size)
		self.encoder_av_to_l = encoderLSTM(self.d_a+self.d_v,zl_size)

		self.decoder_l = decoderLSTM(fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fa_size,self.d_a)
		self.decoder_v = decoderLSTM(fv_size,self.d_v)
	
		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		# zv_nov,za_noa,zl_nol are not activated (mse loss with zl,za,zv)
		zv_nov = self.encoder_la_to_v.forward(torch.cat([x_l,x_a], dim=2))
		za_noa = self.encoder_lv_to_a.forward(torch.cat([x_l,x_v], dim=2))
		zl_nol = self.encoder_av_to_l.forward(torch.cat([x_a,x_v], dim=2))

		mmd_loss = loss_MMD(zv_nov)+loss_MMD(za_noa)+loss_MMD(zl_nol)

		# fl,fa,fv are relu activated
		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl_nol)))))
		fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za_noa)))))
		fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv_nov)))))

		dec_len = t
		x_l_hat_nol = self.decoder_l.forward(fl,dec_len)
		x_a_hat_noa = self.decoder_a.forward(fa,dec_len)
		x_v_hat_nov = self.decoder_v.forward(fv,dec_len)
		
		decoded_nol = [x_l_hat_nol]
		decoded_noa = [x_a_hat_noa]
		decoded_nov = [x_v_hat_nov]

		return decoded_nol,decoded_noa,decoded_nov,mmd_loss

class basic_missing(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(basic_missing, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		zy_size = config['zy_size']
		zl_size = config['zl_size']
		za_size = config['za_size']
		zv_size = config['zv_size']
		fy_size = config['fy_size']
		fl_size = config['fl_size']
		fa_size = config['fa_size']
		fv_size = config['fv_size']
		zy_to_fy_dropout = config['zy_to_fy_dropout']
		zl_to_fl_dropout = config['zl_to_fl_dropout']
		za_to_fa_dropout = config['za_to_fa_dropout']
		zv_to_fv_dropout = config['zv_to_fv_dropout']
		fy_to_y_dropout = config['fy_to_y_dropout']
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		last_mfn_size = total_h_dim + config["memsize"]
		output_dim = config['output_dim']

		self.encoder_la_to_y = encoderLSTM(self.d_l+self.d_a,zy_size)
		self.encoder_lv_to_y = encoderLSTM(self.d_l+self.d_v,zy_size)
		self.encoder_av_to_y = encoderLSTM(self.d_a+self.d_v,zy_size)
	
		self.zy_nol_to_y_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_nol_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.zy_nol_to_y_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zy_noa_to_y_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_noa_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.zy_noa_to_y_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zy_nov_to_y_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_nov_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.zy_nov_to_y_dropout = nn.Dropout(zy_to_fy_dropout)

	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]

		# zv_nov,za_noa,zl_nol are not activated (mse loss with zl,za,zv)
		zy_nov = self.encoder_la_to_y.forward(torch.cat([x_l,x_a], dim=2))
		zy_noa = self.encoder_lv_to_y.forward(torch.cat([x_l,x_v], dim=2))
		zy_nol = self.encoder_av_to_y.forward(torch.cat([x_a,x_v], dim=2))

		mmd_loss = loss_MMD(zy_nov)+loss_MMD(zy_noa)+loss_MMD(zy_nol)

		y_hat_nol = self.zy_nol_to_y_fc2(self.zy_nol_to_y_dropout(F.relu(self.zy_nol_to_y_fc1(zy_nol))))
		y_hat_noa = self.zy_noa_to_y_fc2(self.zy_noa_to_y_dropout(F.relu(self.zy_noa_to_y_fc1(zy_noa))))
		y_hat_nov = self.zy_nov_to_y_fc2(self.zy_nov_to_y_dropout(F.relu(self.zy_nov_to_y_fc1(zy_nov))))
		
		return y_hat_nol,y_hat_noa,y_hat_nov,mmd_loss

