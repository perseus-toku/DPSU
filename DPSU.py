import numpy as np
import torch
import torch.nn.functional as F
import os
import generate_data
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from copy import deepcopy

class PreprocessingValidator(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_transformations, batch_sz):
        # num Transformations defines the number of data transformations we have
        # classification --> task specific
        super(PreprocessingValidator, self).__init__()

        self.fc1 = torch.nn.Linear(num_inputs, 64)
        self.fc2 = torch.nn.Linear(64 ,32)
        self.fc3 = torch.nn.Linear(32,num_outputs)

        self.batch_sz = batch_sz
        self.num_trainsformations = num_transformations

        self.gate_params = {}
        self.gate_weights = {}
        for i in range(num_transformations):
            G = torch.nn.Parameter((torch.ones(1)), requires_grad=True)
            # the parameters have to be explicitly registered
            self.register_parameter(f"G{i}", G)
            self.gate_params[i] = G
            self.gate_weights[i] = None

    def forward(self,x):
        # implement batch size here to speed up training process 

        sigs = []
        norm_factor = torch.zeros(1)
        for i in range(self.num_trainsformations):
            sig = torch.sigmoid(self.gate_params[i])
            sigs.append(sig)
            norm_factor += torch.exp(sig)
        concat_input = []
        for i in range(self.num_trainsformations):
            gate_weight = torch.exp(sigs[i])/norm_factor
            self.gate_weights[i] = gate_weight
            concat_input.append(x[i,:]*gate_weight)

        x  = torch.cat(concat_input, 0)
        # x  = torch.cat((x[0,:]* torch.sigmoid(self.G1),  x[1,:]* torch.sigmoid(self.G2),  x[2,:]* torch.sigmoid(self.G3)), 0 )
        # concatenate the new x together --> gates are optimized this way
        # flatten the input data
        x = x.reshape(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# train with the DPSU
def train_network(train_samples, train_labels, test_samples, test_labels,  NUM_EPOCH=50, batch_sz=50):
    print("start training...")
    # extract the feature sizes and size of each transformations
    # get the number of data transformations
    _, num_transformations, _ = train_samples.shape
    num_input = train_samples[0].numel()
    _, num_output = train_labels.shape

    net = PreprocessingValidator(num_input, num_output, num_transformations, batch_sz=batch_sz)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr= 0.001, weight_decay=1e-5)
    print("######## All parameters")
    for name, param in net.named_parameters():
        print(name, type(param.data), param.size())

    train_process = []
    test_process = []
    ## Train the given neural network
    for e in range(NUM_EPOCH):
        print("Epoch ",e)
        print("-"*10)
        running_loss = 0.0
        # show the values of the three parameters
        print("[A Parameter values]", net.gate_weights)
        for i in range(len(train_samples)):
            optimizer.zero_grad()
            sample, label = train_samples[i], train_labels[i]
            output = net(sample)
            # loss = criterion(output, label)  + 0.2*(torch.abs(net.G1)+torch.abs(net.G2)+torch.abs(net.G3))
            # l2_regularizer = torch.abs(net.G1)+ torch.abs(net.G2)+ torch.abs(net.G3)
            loss = criterion(output, label)
            loss.backward()
            running_loss += loss
            optimizer.step()

            # for name, param in net.named_parameters():
            #     print(name, type(param.data), param.size())
        for k in net.gate_params:
            s = ""
            s += str(net.gate_params[k].grad) + " "
        print(s)

        print("train loss is ",running_loss.tolist())
    print("finished training")
    return net


def generate_inputs_for_func_est(norm=True):
    # generate input data for function estimate task with several different preprocessing techniques
    # apply normalization to make sure input data are consistent
    num_train_samples = 5000
    num_test_samples = 500
    num_variables = 4
    train, train_label, test, test_label = generate_data.generate_data(num_train_samples, num_test_samples, num_variables, data_type="func_addition")

    # apply some data transformations
    def dt_linear(samples):
        samples = deepcopy(samples)
        # basically gives out ground truth --> makes the problem linear
        for i in range(len(samples)):
            samples[i][0] = torch.exp(samples[i][0])
            samples[i][1] = samples[i][1]**2
            samples[i][2] = torch.sqrt(samples[i][2])
            samples[i][3] = torch.sin(samples[i][3]) * samples[i][0]
        return samples

    def dt2(samples):
        samples = deepcopy(samples)
        # Solve the most difficult part
        for i in range(len(samples)):
            samples[i][0] = torch.exp(samples[i][0])
            samples[i][1] = samples[i][1]**2
            samples[i][2] = torch.sqrt(samples[i][2])
            samples[i][3] = torch.sin(samples[i][3])
        return samples

    def df_add_noise(samples, nosie=0.5):
        samples = deepcopy(samples)
        ## add further noise to the sample
        sample_noise = torch.FloatTensor(np.random.normal(size=samples.shape) * nosie + 1)
        samples = samples * sample_noise
        return samples

    dt1_train, dt1_test = dt_linear(train), dt_linear(test)
    dt2_train, dt2_test = dt2(train), dt2(test)
    dt3_train, dt3_test = df_add_noise(train), df_add_noise(test)

    if norm:
        train, test = generate_data.normalize_samples(train), generate_data.normalize_samples(test)
        dt1_train, dt1_test = generate_data.normalize_samples(dt1_train), generate_data.normalize_samples(dt1_test)
        dt2_train, dt2_test = generate_data.normalize_samples(dt2_train), generate_data.normalize_samples(dt2_test)
        dt3_train, dt3_test = generate_data.normalize_samples(dt3_train), generate_data.normalize_samples(dt3_test)

    transformations_train = torch.stack([train, dt1_train, dt2_train, dt3_train], 1)
    transformations_test = torch.cat([test, dt1_test, dt2_test, dt3_test], 1)
    return transformations_train, train_label, transformations_test, test_label


def run_experiemnt():
    # first generate_training_data
    transformations_train, train_label, transformations_test, test_label = generate_inputs_for_func_est(norm=True)
    train_network(transformations_train, train_label, transformations_test, test_label,  NUM_EPOCH=100)





if __name__ == "__main__":
    # run experiemnt
    run_experiemnt()
