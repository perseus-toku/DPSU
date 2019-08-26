import numpy as np
import torch
import torch.nn.functional as F
import os
from torchvision import transforms


############ create all data trasnformatations and generate training data based on the transfroamtions
# basic input data is from a normal distribution
# ideal property to have is from X~Y and Y has certain variabilities
def generate_a_random_input(arr_len):
    arr = np.random.rand(1,arr_len)
    return arr


def generate_k_samples(k, sample_length):
    samples =[]
    for i in range(k):
        samples.append(generate_a_random_input(sample_length))
    return np.array(samples)


def ground_truth_trainsformation(s):
    """x is a sample input vector and the ground truth transformations
        applies a chain of transformations to the underlying data

        ret: the label for this given x
    """
        # step 1: exp
    s = np.exp(s)
    # step 2: square
    s = s*s
    #step 3: sum
    s = np.sum(s)
    # alternatively can do sine wave estimation
    # exp sine estimation
    return s


def generate_labels(samples):
    # apply a given function to the input samples
    labels = []
    for s in samples:
        label = ground_truth_trainsformation(s)
        labels.append(label)
    return labels


def generate_data(num_train, num_test, sample_length, data_type="Linear"):
    params = torch.FloatTensor([[1],[2],[10],[15]])

    mu, sigma = 5, 1
    train = torch.FloatTensor(abs(np.random.normal(loc=mu, scale=sigma, size=(num_train, sample_length))))
    test = torch.FloatTensor(abs(np.random.normal(loc=mu, scale=sigma, size=(num_test, sample_length))))
    # generate a noise term
    train_noise = torch.FloatTensor(np.random.normal(size=train.shape) * 1e-3 + 1)
    test_noise = torch.FloatTensor(np.random.normal(size=test.shape) * 1e-3 + 1)
    noise_train = train * train_noise
    noise_test = test * test_noise

    if data_type == "Linear":
        # this task is to make the network learn a function F: = w1*exp(x1) + w2*exp(x2) + ... + wn*exp(xn)
        # for data in the positive range
        # this task is to make the network learn a function F: = w1*exp(x1) + w2*exp(x2) + ... + wn*exp(xn)
        train_label = torch.mm(noise_train, params)
        test_label = torch.mm(noise_test, params)

    elif data_type == "Exponential":
        # for data in the positive range
        # this task is to make the network learn a function F: = w1*exp(x1) + w2*exp(x2) + ... + wn*exp(xn)
        train_label = torch.FloatTensor(torch.exp(noise_train))
        train_label = torch.mm(train_label, params)

        test_label = torch.FloatTensor(torch.exp(noise_test))
        test_label = torch.mm(test_label, params)

    elif data_type == "exp+square":
        # f = W*exp(X^2)
        train_label = torch.FloatTensor(torch.exp(noise_train)**2)
        train_label = torch.mm(train_label, params)

        test_label = torch.FloatTensor(torch.exp(noise_test)**2)
        test_label = torch.mm(test_label, params)

    elif data_type == "func_addition":
        # there are four paras
        def data_pipeline(samples):
            for i in range(len(samples)):
                samples[i][0] = torch.exp(samples[i][0])
                samples[i][1] = samples[i][1]**2
                samples[i][2] = torch.sqrt(samples[i][2])
                # non-linear interactions between input data 
                samples[i][3] = torch.sin(samples[i][3]) * samples[i][0]
            return samples

        train_label = data_pipeline(noise_train)
        test_label = data_pipeline(noise_test)
        train_label = torch.mm(train_label, params)
        test_label = torch.mm(test_label, params)

    return train, train_label, test, test_label


def normalize_samples(X):
    means = X.mean(dim=0)
    stds = X.std(dim=0)
    X = X - means
    X = X/stds
    return X


if __name__ == "__main__":
    train, train_label, test, test_label = generate_data(10, 5, 4, "func_addition")
    normalize_train_samples(train)
