from fastai.vision.all import *
from fastbook import *
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

def mse(pred, targets): 
    return ((pred-targets)**2).mean().sqrt()

def mclrn(x, param):
    a,b,c,d,e = param
    return (a - b*(x**2/math.factorial(2)) + c*(x**4/math.factorial(4)) - d*(x**6/math.factorial(6)) + e*(x**8/math.factorial(8)))

def apply_step(x,y, param, lr):
    preds = mclrn(x, param)
    loss = mse(preds, y)
    loss.backward()
    param.data -= lr*param.grad.data
    param.grad = None
    return preds.detach(), loss.item()
    

def plot_in_realtime(x, y, preds, losses, trn_tm, params):
    plt.clf()  # Clear the current figure
    plt.subplot(1, 2, 1)
    plt.plot(x.numpy(), preds.numpy(), ".", label="Predictions")
    plt.plot(x.numpy(), y.numpy(), "+", label="Targets")
    plt.legend(fontsize='small')
    #plt.text(0.5, 0.95, f'params: {params.data}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


    plt.subplot(1, 2, 2)
    plt.plot(losses,label=f"Loss")
    plt.xlabel("Iteration", fontsize=6)
    plt.ylabel("Loss", fontsize=6)
    plt.legend(fontsize='small')
    plt.xlim([-1000,max(trn_tm)])
    
    current_loss = losses[-1] if losses else 0
    plt.text(0.5, 0.95, f'Loss: {current_loss:.4f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.pause(0.001)  # Pause to update the plots


def main():
    
    lr = 1e-4
    
    Noise_level = 0.2
    noise = torch.randn(100)*Noise_level

    x = torch.linspace(-2*torch.pi,2*torch.pi, steps= 100) 
    y = torch.cos(x) + noise
    params = torch.randn(5).float().requires_grad_()

    losses = []
    training_time = range(50000) 
    for i in training_time: 
        preds, loss = apply_step(x, y, params, lr)
        losses.append(loss)
        if i < 2100:
                if i % 10 == 0:
                    plot_in_realtime(x, y, preds, losses, training_time, params)
        elif i % 500 == 0:  # Update the plot every 10 iterations
            plot_in_realtime(x, y, preds, losses, training_time, params)
    
    plt.ioff()  # Turn off interactive mode
    plt.show() 


if __name__ == "__main__":
    main()