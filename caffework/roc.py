import caffe
#import lmdb
import os
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from sklearn import metrics

# load the net with trained weights
caffe.set_mode_gpu()
caffe.set_device(7)
#caffe.set_phase_test()

net = caffe.Net('fall_bnn.prototxt', 'bnn_train_iter_5000.caffemodel', caffe.TEST)

dirname = os.path.abspath('../h5data')
test_filename = os.path.join(dirname, 'test.h5')
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'r') as f:
    for key in f.keys():
        print(key)
    label = f['label']
    dat = f['data'][:]

    obs_dims = dat.shape
    print(obs_dims)
    
    for i in range(len(dat)):
        net.blobs['data'].data[...] = dat[i]
        if label[i] == 1: 
            #print(dat[i])
            #print(dat[i].shape)
            print("label",label[i])
        
        output = net.forward()  # get prediction for x_i

        if label[i] == 1: 
            print("accuracy",output['accuracy'])    
        #print('output: ',output['prob'][0][0])    

#lmdb_file = "../val_lmdb"

#print("Open val lmdb file")
#lmdb_env = lmdb.open(lmdb_file)

#lmdb_txn = lmdb_env.begin()
#lmdb_cursor = lmdb_txn.cursor()
#datum = caffe_pb2.Datum()
y_score = []
y_true = []



quit()



print("Processing...")
n_samples=0
noFall = 0
fall = 0


with lmdb_env.begin() as lmdb_txn:
    with lmdb_txn.cursor() as lmdb_cursor:
        for key, value in lmdb_cursor:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            y_i = datum.label
            x_i = np.fromstring(datum.data).reshape(
                 datum.channels, datum.height, datum.width) # get i-th validation sample
            y_true.append( y_i )  # y_i is 0 or 1 the TRUE label of x_i
            output = net.forward( data=x_i )  # get prediction for x_i
#            output = net.forward(data=np.asarray(x_i))
            #print(output['loss'][0][0])
#            print('x_i:',x_i,', y_i:',y_i,', output:',output,'\n\n')

            if y_i == 0:
                noFall = noFall + 1
                #print('false (no fall)')
            elif y_i == 1:
                fall = fall + 1
                #print('true (fall)')
            else:
                print('undefined!!!')
            y_score.append( output['prob'][0][0] ) # get score for "1" class
            # once you have N y_score and y_true values
            n_samples=n_samples+1
#            print('sample: ',n_samples,', y_true: ',y_i,', y_score: ',output['prob'][0][0])
#            print('key: ',key)
#            print('value: ',value)
#            print('datum: ',datum)
            print('x_i: ',x_i)
            print('output: ',output)
#            print('output[loss]: ',output['loss'])
#            print('output[accuracy]: ',output['accuracy'])
#            print('output[prob]: ',output['prob'])
#            print('output[prob][0][0]: ',output['prob'][0][0])
#            break

            
#print('y_true:',y_true,'\n\n');
#print('y_score:',y_score,'\n\n');
#fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
#auc = metrics.roc_auc_score(y_true, y_score)
auc = metrics.roc_auc_score(fpr, tpr)
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://gist.github.com/bearpaw/3a07f0e8904ed42f376e
# https://shengshuyang.github.io/hook-up-lmdb-with-caffe-in-python.html

print ('num samples:',n_samples)
print ('num falls:',fall)
print ('num no falls:',noFall)
print ('tpr',tpr)
print ('fpr',fpr)
print ('thresholds',thresholds)
print ('auc',auc)

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc )
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Fall Detection Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.png')
