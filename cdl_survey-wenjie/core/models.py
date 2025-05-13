import torch
import torch.nn as nn
import tensorflow as tf

class MLP(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers=4, layer_width=100,bn=False,out_activation ='Sigmoid'):
        super(MLP, self).__init__()
        #MLP model parameters
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_features if i==0 else layer_width, layer_width))
            if bn:
                layers.append(nn.BatchNorm1d(layer_width))
            layers.append(nn.ReLU())
            
            
        layers.append(nn.Linear(layer_width, num_classes))
        
        if out_activation == 'Sigmoid':
            layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)
        return x
    
def create_tf_model(num_classes,num_features,num_layers,layer_width=100,bn=False,out_activation='sigmoid'):
    layers = []
    layers.append(tf.keras.Input(shape=(num_features,)))
    for _ in range(num_layers-1):
        layers.append(tf.keras.layers.Dense(layer_width))
        if bn:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation('relu'))
    layers.append(tf.keras.layers.Dense(num_classes,activation=out_activation.lower())) #,activation='tanh'
    model = tf.keras.Sequential(layers)
    return model
