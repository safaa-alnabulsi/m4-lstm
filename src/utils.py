import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize(train_dataset):
    scaler = StandardScaler()
    scaler.fit(train_dataset)
    return scaler.transform(train_dataset)

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def plot_samples(dataframe, num_of_samples):
    samples = dataframe.iloc[:num_of_samples]
    x = np.arange(0,samples.shape[1])
    plt.figure(figsize=(20,10))
    plt.plot(x, samples.T.values)


def univariate_data(dataset, start_index, end_index):
    data = []
    labels = []

    data   = dataset[:, start_index:end_index]
    labels = dataset[:, end_index:]
    
    return np.array(data), np.array(labels)

def create_time_steps(length, start):
    time_steps = []
    for i in range(start, length + start):
        time_steps.append(i)
        
    return time_steps

def show_plot(history, future, start_future, horizon_len,title='',model_prediction=None):
    #print(start_future, horizon_len)
    time_steps_history = create_time_steps(start_future, start=0)
    time_steps_future  = create_time_steps(horizon_len , start=start_future)

    plt.figure(figsize=(20,10))
    plt.title(title)
    
    #plt.plot(time_steps_history, history, '-', markersize=10, color = 'blue', label='History')
    plt.plot(time_steps_future , future , '-', markersize=10, color = 'red' , label='True Future')
    if model_prediction is not None:
        plt.plot(time_steps_future, model_prediction, '-', markersize=10, color = 'green', label='Model Prediction')

    plt.legend()
    plt.xlabel('Time-Step')
    plt.ylabel('Audience Size')
    
    return plt