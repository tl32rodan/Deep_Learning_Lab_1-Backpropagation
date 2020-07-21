from gen_data import *
from TLNN import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def acc(ground_truth, y):
    print ('Accuracy = ',100* (ground_truth[ground_truth==y].size / ground_truth.size) , '%')

def run_Net(data_x, data_y, num_epochs = 10000, batch_size=20, print_freq = 1000, lr=0.05, plot_loss_freq=10, enable_loss_msg = True, layer_1_units = 4, layer_2_units = 4):
    net = TLNN(layer_1_units, layer_2_units)
    data_size = data_x.shape[0]
    iter_per_epoch = max(round(data_size/batch_size), 1)
    
    loss_list = {'loss':[],'epochs':[]}
    
    for i in range(num_epochs):
        # Feed data to network in batch
        loss_sublist = []
        for j in range(iter_per_epoch):
            # Get mini-batch
            batch_mask = np.random.choice(data_size, batch_size)
            X, y = data_x[batch_mask], data_y[batch_mask]

            predict = net.forward(X)
            loss = net.cal_loss(predict,y)
            loss_sublist.append(loss)
            net.backward(lr=lr)

        avg_loss = np.mean(loss_sublist)
        # 存 loss
        if (i+1) % plot_loss_freq == 0:
            loss_list['loss'].append(avg_loss)
            loss_list['epochs'].append(i)
        
        if enable_loss_msg and (i+1) % print_freq == 0:
            print('epoch {} loss: {}'.format(i+1, avg_loss))
            
    return net.forward(data_x), loss_list

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize= 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize= 18)
    for i in range(x.shape[0]):
        if pred_y[i]< 0.5 :
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.show()

    