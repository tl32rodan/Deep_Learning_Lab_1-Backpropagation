from gen_data import *
from run_net import *
    
if __name__ == '__main__':
    # XOR data
    data_x, data_y = generate_XOR_easy()
    num_epochs = 500000
    #num_epochs = 10000
    batch_size= 5
    print_freq = 10000
    lr=0.01
    plot_loss_freq=200
    
    pred_y, loss_list = run_Net(data_x,data_y,num_epochs, batch_size, print_freq, lr, plot_loss_freq,layer_1_units = 4, layer_2_units = 4)
    
    # Comparison graph
    show_result(data_x,data_y,pred_y)
    
    # Show accuracy
    pred_y[pred_y>0.5] = 1
    pred_y[pred_y<0.5] = 0
    acc(data_y,pred_y)
    
    # loss/epoch curve
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(loss_list['epochs'],loss_list['loss'])
    plt.savefig('result_train_XOR.png')