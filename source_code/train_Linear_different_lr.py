from gen_data import *
from run_net import *
from TLNN import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Linear data
    data_x, data_y = generate_linear(n=100)

    num_epochs = 2000
    batch_size=20
    print_freq = 500
    #lr=0.1
    plot_loss_freq=10

    plt.figure(figsize=(18,12))
    plt.title('Train with different learning rate', fontsize= 18)

    for i, lr in enumerate([1, 0.75, 0.5, 0.1, 0.05, 0.01],1):
        pred_y, loss_list = run_Net(data_x,data_y,num_epochs, batch_size, lr, plot_loss_freq, enable_loss_msg = False)

        # Comparison graph
        #show_result(data_x,data_y,pred_y)

        # Show accuracy
        pred_y[pred_y>0.5] = 1
        pred_y[pred_y<0.5] = 0
        #acc(data_y,pred_y)

        # loss/epoch curve
        plt.subplot(2,3,i)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(loss_list['epochs'],loss_list['loss'], label=str(lr))
        plt.legend()
    plt.show()
