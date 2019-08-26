import numpy as np
import torch
import torch.nn.functional as F
import os
import generate_data
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import matplotlib.pyplot as plt
from termcolor import colored

class Net(torch.nn.Module):
    """ baseline multi-layer perceptron model
    """
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(num_inputs,64)
        self.fc2 = torch.nn.Linear(64,32)
        self.fc3 = torch.nn.Linear(32,num_outputs)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def run_test_with_net(test_samples, test_labels, net, criterion, batch_sz=20):
    # evaluate on test set
    running_test_loss =0.0
    sample_size, _ = test_samples.shape
    for i in range(0, sample_size, batch_sz):
        sample, label = test_samples[i:i+batch_sz], test_labels[i:i+batch_sz]
        output = net(sample)
        loss = criterion(output, label)
        running_test_loss += loss
    running_test_loss = running_test_loss/sample_size

    def visualize_random_test():
        ind = np.random.randint(0,sample_size-1)
        # visulize some test outputs
        random_test = test_samples[ind]
        output = net(random_test)
        ground_truth = test_labels[ind]
        print(output, ground_truth)
    visualize_random_test()

    return running_test_loss


# training is done for simple transformation functions to prove the efficiency of the function estimation
def train_network(train_samples, train_labels, test_samples, test_labels, NUM_EPOCH=50, batch_sz=20, save_model_idx=0, read_model_idx=None):
    print("start training...")
    writer = SummaryWriter('runs/exp-1')

    # get input and output size from the train_samples
    sample_size, num_variables = train_samples.shape
    # get the output size
    _, output_size = test_labels.shape

    net = Net(num_variables, output_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr= 0.001, weight_decay=1e-5)

    if read_model_idx:
        print("Loading an existing model with idx=", read_model_idx)
        assert os.path.exists("models"), "no saved model found"
        path = os.path.join("models", str(read_model_idx))
        net.load_state_dict(torch.load(path))

    train_errors = []
    test_errors = []
    running_loss = 0.0
    running_test_loss = 0.0
    ## Train the given neural network
    for e in range(NUM_EPOCH):
        print(colored(f"Epoch {e}", 'blue'))
        print("-"*40)
        running_loss = 0.0
        # load based on the batch_sz
        for i in range(0, len(train_samples), batch_sz):
            optimizer.zero_grad()
            sample, label = train_samples[i:i+batch_sz], train_labels[i:i+batch_sz]
            output = net(sample)
            loss = criterion(output, label)
            loss.backward()
            running_loss += loss
            optimizer.step()

        running_loss = running_loss/sample_size
        running_test_loss = run_test_with_net(test_samples, test_labels, net, criterion, batch_sz)

        writer.add_scalar('Train/Loss', running_loss, e+1)
        writer.add_scalar('Test/Loss', running_test_loss, e+1)

        # print out the error for each epoch training
        train_errors.append(running_loss)
        test_errors.append(running_test_loss)
        print(colored( f"train loss is {running_loss.tolist()} test loss is {running_test_loss.tolist()}", 'green'))
    print("finished training")

    # save the model
    if not os.path.exists("models"):
        os.mkdir("models")
    path = os.path.join("models", str(save_model_idx))
    torch.save(net.state_dict(),path)
    return net

def test_with_simple_models(norm=False):
    num_train_samples = 5000
    num_test_samples = 500
    num_variables = 4
    train, train_label, test, test_label = generate_data.generate_data(num_train_samples,
     num_test_samples, num_variables, data_type="func_addition")

    if norm:
        train = generate_data.normalize_samples(train)
        test = generate_data.normalize_samples(test)

    net = train_network(train, train_label, test, test_label, NUM_EPOCH=500, batch_sz=50)

def visulize_error_rate(net=None, norm=False):
    # this proves the network can properly learn the exponential function
    num_train_samples = 100
    num_test_samples = 5
    num_variables = 4
    net = Net(4, 1)

    read_model_idx = 0
    path = os.path.join("models", str(read_model_idx))
    net.load_state_dict(torch.load(path))
    train, train_label, test, test_label = generate_data.generate_data(num_train_samples, num_test_samples,
     num_variables, data_type="func_addition")

    if norm:
        train = generate_data.normalize_samples(train)
        test = generate_data.normalize_samples(test)
    X = []
    Y = []

    for i, s in enumerate(train):
        output = net(s).item()
        label = train_label[i].item()
        X.append(i)
        error = (output-label)/label
        Y.append(error)

    print(f"avg error is ")
    plt.plot(X,Y)
    plt.show()



if __name__ == "__main__":
    test_with_simple_models(norm=True)
    visulize_error_rate(norm=True)
