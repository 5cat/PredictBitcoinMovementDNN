import pandas as pd 
import numpy as np 
import datetime
from sklearn.utils.class_weight import compute_sample_weight
from termcolor import colored
from tqdm import tqdm
np.random.seed(1)

DISTANCE=60
TIMESTEP_SIZE=60
BATCH_SIZE=256
TEST_RATIO=0.25

PRINT_DATA_EACH=50
TEST_EACH=PRINT_DATA_EACH*10
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

df=pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv")

split_date="01/01/2017"
split_timestamp=datetime.datetime.strptime(split_date, "%d/%m/%Y").timestamp()

data=[]

null_data=[]
nnull_data=[]
c=0
last_d=None
for d in df.values:
	if d[0]>=split_timestamp and (not np.isnan(d[4])):
		if np.isnan(d[4]):
			data.append(last_d[4])
		else:
			last_d=d
			data.append(d[4])


split_index=int(len(data)*TEST_RATIO)
train_data=np.array(data[:-split_index])
test_data=np.array(data[-split_index:])


def get_x(data,start_index):
	sampled_x=np.expand_dims(data[start_index:start_index+TIMESTEP_SIZE],axis=1)
	sampled_x=(sampled_x/(sampled_x[0]+1e-8))-1
	return sampled_x

def get_y(data,start_index):
	seq=data[start_index+TIMESTEP_SIZE:start_index+TIMESTEP_SIZE+DISTANCE]
	st_p=seq[0]
	y_ = np.mean(seq) / st_p
	return int(y_>1)


ys_train_data_is={0:[],1:[]}
for i in tqdm(range(len(train_data)-TIMESTEP_SIZE-DISTANCE),desc='ys_train_data_is'):
	y_=get_y(train_data,i)
	ys_train_data_is[y_].append(i)

total_=sum(map(len,ys_train_data_is.values()))
print(" , ".join(["{}:{:.5%}".format(key,len(value)/total_) for key,value in ys_train_data_is.items()]))


def generator(data,TIMESTEP_SIZE,BATCH_SIZE):

	while True:
		start_index_0=np.random.choice(ys_train_data_is[0],size=BATCH_SIZE//2) #np.random.randint(0,len(data)-TIMESTEP_SIZE-DISTANCE)
		start_index_1=np.random.choice(ys_train_data_is[1],size=BATCH_SIZE//2)
		x=[]
		y=[]
		for start_index in np.append(start_index_0,start_index_1):
			sampled_y=get_y(data,start_index)
			sampled_x=get_x(data,start_index)
			x.append(sampled_x)
			y.append(sampled_y)

		yield np.array(x),np.array(y), np.ones(len(y))#get_sample_weight(y)


def get_sample_weight(y):
	return compute_sample_weight("balanced",y)

from keras import backend as K

config = K.tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = K.tf.Session(config=config)
K.set_session(sess)

from keras.models import Model,load_model
from keras import layers as kl 
from keras import optimizers
from keras import backend as K


def average_pred(y_true,y_pred):
	return K.mean(y_pred)


li=kl.Input(shape=(TIMESTEP_SIZE,1))
l=li
for n_units in [16,32,64,128]:
	l=kl.Conv1D(n_units,3,activation='elu',padding='same')(l)
	l=kl.MaxPooling1D()(l)

l=kl.Flatten()(l)
l=kl.Dense(64,activation='elu')(l)
l=kl.Dense(1,activation='sigmoid')(l)

model=Model(li,l)
model.compile(optimizers.Adamax(lr=0.002),"binary_crossentropy",['acc',average_pred])


if not args.test:
	from keras.callbacks import TensorBoard
	import os
	iteration=0
	res=[]
	print_msg="iteration: {} : loss: {:.6f}, acc: {:.4%}, avg_pred: {:.4f}, avg_y: {:.4f}, left_iter_to_test: {}"
	best_score=np.inf
	for filename in os.listdir("./logs"):
		os.remove("./logs/{}".format(filename))
	tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,batch_size=BATCH_SIZE,
						  write_graph=True, write_images=False)
	tensorboard.set_model(model)

	for x,y,w in generator(train_data,TIMESTEP_SIZE,BATCH_SIZE):
		r=model.train_on_batch(x,y,w)
		tensorboard.on_epoch_end(iteration,{'train_loss':r[0],'train_acc':r[1]})
		r+=[np.mean(y)]
		res.append(r)

		iteration+=1
		if iteration%PRINT_DATA_EACH==0:
			print(print_msg.format(iteration,*np.mean(res,axis=0), TEST_EACH-((iteration-1)%TEST_EACH)))
			res=[]

		if iteration%(TEST_EACH)==0:
			true=[]
			test=[]
			for i in tqdm(range(len(test_data)-TIMESTEP_SIZE-DISTANCE)):
				test.append(get_x(test_data,i))
				true.append(get_y(test_data,i))
			true,test=np.array(true),np.array(test)
			pred=model.evaluate(test,true,sample_weight=get_sample_weight(true),verbose=1,batch_size=BATCH_SIZE)
			msg=''
			if best_score>pred[0]:
				best_score=pred[0]
				model.save("model.h5")
				msg+=", FOUND A NEW SCORE"
			tensorboard.on_epoch_end(iteration,{'test_loss':pred[0],'test_acc':pred[1]})
			print(colored("res: {}{}".format(np.array(pred),msg),"blue","on_white",['bold']))


model.load_weights("model.h5")

true=[]
test=[]

for i in tqdm(range(len(test_data)-TIMESTEP_SIZE-DISTANCE),desc="loading testing data"):
	test.append(get_x(test_data,i))
	true.append(get_y(test_data,i))


pred=model.predict(np.array(test),verbose=1)


def get_accp(x,true,pred):
	s=[]
	for t,p in zip(true,pred):
		if p>x:
			p=int(p+0.5)
			s.append(p==t)

	return np.mean(s),len(s)/len(true)

xs=[]
ys=[]
zs=[]
for i in np.arange(0.5,1,0.01):
	xs.append(i)
	r,n=get_accp(i,true,pred)
	ys.append(r)
	zs.append(n)
	if n==0:
		break
	print("pred: {:.2f}, acc: {:.4%}, occurrence: {:.6%}".format(i,r,n))

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('predicted probability')
ax1.set_ylabel('occurrence', color=color)
ax1.plot(xs, zs, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel(' acc', color=color)
ax2.plot(xs, ys, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()