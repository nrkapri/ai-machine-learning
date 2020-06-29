# Artiicial Neural Network 
An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Here, each circular node represents an artificial neuron and an arrow represents a connection from the output of one artificial neuron to the input of another.

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/560px-Colored_neural_network.svg.png)

#### Neuron 

ANNs are composed of artificial neurons which retain the biological concept of neurons, which receive input, combine the input with their internal state (activation) and an optional threshold using an activation function, and produce output using an output function. The initial inputs are external data, such as images and documents. The ultimate outputs accomplish the task, such as recognizing an object in an image. The important characteristic of the activation function is that it provides a smooth, differentiable transition as input values change, i.e. a small change in input produces a small change in output.

Basic structure
For a given artificial neuron, let there be m + 1 inputs with signals x0 through xm and weights w0 through wm. Usually, the x0 input is assigned the value +1, which makes it a bias input with wk0 = bk. This leaves only m actual inputs to the neuron: from x1 to xm.

The output of the kth neuron is:

![Image](https://wikimedia.org/api/rest_v1/media/math/render/svg/be21980cc9e55ea0880327b9d4797f2a0da6d06e)

Where ![Image](https://wikimedia.org/api/rest_v1/media/math/render/svg/33ee699558d09cf9d653f6351f9fda0b2f4aaa3e) (phi) is the transfer function (commonly a threshold function).

![Image](https://upload.wikimedia.org/wikipedia/commons/b/b0/Artificial_neuron.png)

The output is analogous to the axon of a biological neuron, and its value propagates to the input of the next layer, through a synapse. It may also exit the system, possibly as part of an output vector.

It has no learning process as such. Its transfer function weights are calculated and threshold value are predetermined.


#### Activation Function

#### Gradient Descent

#### Stochastic Gradient Descend 

#### Backpropagation 

