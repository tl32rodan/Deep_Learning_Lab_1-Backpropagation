from gen_data import *
from run_net import *
    
if __name__ == '__main__':
    # XOR data
    data_x, data_y = generate_XOR_easy()
    num_epochs = 500000
    batch_size=20
    print_freq = 50000
    lr=0.025
    plot_loss_freq=200
    
    pred_y, loss_list = run_Net(data_x,data_y,num_epochs, batch_size, print_freq, lr, plot_loss_freq)
    
    # Comparison graph
    show_result(data_x,data_y,pred_y)
    
    # Show accuracy
    pred_y[pred_y>0.5] = 1
    pred_y[pred_y<0.5] = 0
    acc(data_y,pred_y)
    
    # loss/epoch curve
    sns.relplot(x="epochs", y="loss", kind='line' ,data=pd.DataFrame(loss_list))
