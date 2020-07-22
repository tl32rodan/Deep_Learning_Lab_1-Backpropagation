from gen_data import *
from run_net import *
import numpy as np
    
if __name__ == '__main__':
    np.random.seed(0)
    # XOR data
    data_x, data_y = generate_XOR_easy()
    num_epochs = 50000
    #num_epochs = 10000
    batch_size= 21
    print_freq = 1000
    
    #lr=0.001490626
    lr=0.001490625
    plot_loss_freq=200
    
    pred_y, loss_list = run_Net(data_x,data_y,num_epochs, batch_size, print_freq, lr, plot_loss_freq,layer_1_units = 4, layer_2_units = 4, act_f = 'relu', loss_f = 'bce')
    
    print('pred_y = ',pred_y )
    # Comparison graph
    show_result(data_x,data_y,pred_y)
    
    # Show accuracy
    pred_y[pred_y>0.5] = 1
    pred_y[pred_y<0.5] = 0
    acc(data_y,pred_y)
     
    # loss/epoch curve
    plt.figure()
    plt.title('Train with ReLU & Binary Cross-Entropy, lr = 0.001490625')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(loss_list['epochs'],loss_list['loss'])
    plt.savefig('result_train_XOR_relu_bce_626.png')