Here's the architecture for a CNN that is used for this task:

Input Layer: This layer will take the 9 grey level images as input. The input shape will be (n_samples, height, width, channels), where n_samples is the number of samples in the dataset, height and width are the dimensions of the images, and channels is set to 1 as the images are grayscale.

Convolutional Layers: We can add multiple convolutional layers to the network to learn different features from the images. Each convolutional layer will have a set of filters (also called kernels) that are used to convolve over the input images and generate a feature map. Each filter will learn to detect a specific feature from the input image. We can use a combination of 2D convolutional layers and pooling layers to reduce the dimensions of the feature maps and extract more high-level features.

Flatten Layer: The output from the convolutional layers will be a 3D tensor, and we need to convert it to a 1D tensor to connect it with the fully connected layers. The Flatten layer will flatten the 3D tensor into a 1D tensor.

Fully Connected Layers: These layers are used to perform the classification task. We can add one or more fully connected layers to the network, with the last layer having n nodes, where n is the number of classes in the dataset. We can also add dropout layers to avoid overfitting.

Output Layer: The output layer will use a softmax activation function to produce a probability distribution over the n classes.

We define a Sequential model and add the layers to it. We use three convolutional layers with increasing number of filters, followed by max pooling layers to reduce the dimensions of the feature maps. We then add a Flatten layer and two fully connected layers with a dropout layer in between them. The last layer has n_classes nodes with softmax activation function.

Finally, we compile the model with Adam optimizer and categorical cross-entropy loss function. We train the model for 10 epochs using a batch size of 32, and we also use a validation set to monitor the model's performance during training.