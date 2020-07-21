from gen_data import *
from run_net import *
    
if __name__ == '__main__':
    # Linear data
    data_x, data_y = generate_linear(n=100)
    
    num_epochs = 10000
    batch_size=20
    print_freq = 500
    lr=0.1
    plot_loss_freq=10
    
    pred_y, loss_list = run_Net(data_x,data_y,num_epochs, batch_size, print_freq, lr, plot_loss_freq)
    
    # Comparison graph
    show_result(data_x,data_y,pred_y)
    
    # Show accuracy
    pred_y[pred_y>0.5] = 1
    pred_y[pred_y<0.5] = 0
    acc(data_y,pred_y)
    
    # loss/epoch curve
    sns.relplot(x="epochs", y="loss", kind='line' ,data=pd.DataFrame(loss_list))
