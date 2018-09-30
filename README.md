 ### **Project Summary**
 The objective of this project is to follow a target with a quadrotor drone in a virtual simulator through the use of semantic segmentation and deep learning. This will be implemented by constructing a Fully Convolutional Neural Network, and the target following accuracy will be calculated using Intersection Over Union metric.
### **Data**
The Drone Simulator in Unity was relatively easy to use, and collect data of the environment, the target, and other similar pedestrians. While I collected data that could be used for the training set, I decided to start with the sample that was provided by Udacity to determine the accuracy that could be achieved. This data can be found [here](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/tree/master/data/train).
### **Network Architecture**
Typically when a convolutional layer is flattened from a 4D to a 2D tensor, it results in the loss of spatial information. In this project, we must not only classify our target, but also retain all pixel location information as well. Therefore, we can implement a 1x1 convolution to preserve this information. 

Furthermore, adding a 1x1 convolution amongst other convolutional layers is a computationally inexpensive way to make the model deeper and allow more parameters, without destroying their structure.

##### **Design Inputs**
 The three most important aspects of our design consist of:
 - **Filter/Kernel Size ( F ):** The size of the window that moves over the input image
 - **Stride ( S ):** The number of steps that the filter window moves over the input image. A stride of 1 would equate to moving the filter one pixel at a time
 - **Zero Padding ( P ):** Place zeros around the border of the image to prevent losing image information, as well as control the spatial size of the output volume
 
##### **Filter/Kernel Size Selection**
Our original image is 256x256 in 2D space, with a depth of 3 for RGB (Red, Blue, Green). This will be downsized to a final input image size of 160x160x3. The number of filters for each layer was determined by using the powers of 2 popular approach:
   | Layer | Purpose | Power | Filter |
| ------ | ------ | ------ | ------ |
| Input | Original Image | - - | 3 |
| 1 | Encoder | 2**6 | 64 |
| 2 | Encoder | 2**7 | 128 |
| 3 | 1x1 Convolution Layer | 2**8 | 256 |
| 4 | Decoder | 2**7 | 128 |
| 5 | Decoder | 2**6 | 64 |
| Output | Decoder | - - | 3 |

These critical parameters were selected by referencing [this](http://cs231n.github.io/convolutional-networks/) helpful resource, as well as adapting popular selections from one of the top performing Convolutional Networks, ResNet.
- **Stride = 2:** A stride of 2 moves the filter 2 pixels at a time, which produces small spatial volumes with a nearly negligible loss of quality
 - **Zero Padding = 1** A minimal border of zeros surrounding the image prevents losing image information, as well as controls the spatial size of the output volume

To determine the convolution layer height and width we use the equation: 
**(W − F + 2 * P) / S + 1**

Therefore the height and width of our first layer will be **80x80x64**: 
**(160 − 3 + 2 * 1) / 2 + 1**

We now have our Network Architecture defined, and will begin to construct our 5 layer FCN, with the 1x1 convolution layer in the middle. It will look something like this:
 ![Network_Arch_FCN](/assets/Network_Arch_FCN.PNG)
 
 ### **FCN Model**
Now that we have properly defined the Architecture and sizing of our FCN, so the next step is to start to implement it! This will be composed of the input image, 2 Encoding layers, a 1x1 convolutional layer, 2 Decoding layers, and finally the ouput. It is important to maintain symmetry by including the same number and sizing of the Encoding and Decoding layers. The FCN Model can be seen here: 
![FCN_Model](/assets/FCN_Model.PNG)

  ##### **Encoder Block**
We begin with defining a separable 2D convolution function that utilizes the Keras library:
```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```
Next, we call this function, and add our parameters for the filter and strides for the input layer:
 ```
 def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```
##### **Encoder Layers**
Now that the Encoder functions are defined, implementing the layers is as simple as calling our encoder block function fore each corresponding layer:
```
def fcn_model(inputs, num_classes):
    # Encoder Blocks. 
    layer_1 = encoder_block(inputs, 64, 2)
    layer_2 = encoder_block(layer_1, 128, 2)
```
 ##### **1x1 Convolution Layer**
For the 1x1 convolution layer we will use a separate 2D convolutional function from the Keras library:
```
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```
Similarly to the Encoding layers, we construct the 1x1 convolutional layer by calling this function and providing the input layer as well as the kernal and stride parameters:
```
    # 1x1 Convolution layer.
    layer_3 = conv2d_batchnorm(layer_2, 256, kernel_size=1, strides=1)
```
 ##### **Decoder**
Similar to the Encoding layers, but in opposite direction, we will begin to Decode our layers to gradually transition from the 1x1x256 to the desired output image size of 160x160x3. A critical element of the Decoding process is the Bilinear Upsampling, which helps reconstruct our downsampled image:
 ```
 def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
 ```
##### **Decoder Blocks**
 We will now define the function to perform Bilinear Upsampling, concatenate the layers, and similarly to the Encoder, create the output layer by calling the separable 2D convolution function provided by the Keras library:
 ```
 def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsample = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concatenate = layers.concatenate([upsample, large_ip_layer])
    
    # Add some number of separable convolution layers
    sep_conv = separable_conv2d_batchnorm(concatenate, filters, 1)
    output_layer = separable_conv2d_batchnorm(sep_conv, filters, 1)
    
    return output_layer
```
##### **Decoder Layers**
Once the Decoder blocks are defined, it's as simple as calling these functions and providing the layers and parameters:
```
    layer_4 = decoder_block(layer_3, layer_1, 128)
    layer_5 = decoder_block(layer_4, inputs, 64)
```
##### **Output Layer**
Finally, we're able to construct the output layer with the same desired sizing as the input image:
```
    # Output Layer
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(layer_5)
```
 ### **Training Parameter Selection**
 ##### **Train**
 Training setup
 ```
 image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

output_layer = fcn_model(inputs, num_classes)
 ```
 ##### **Hyper Parameter Definition**
- **batch_size:** number of training samples/images that get propagated through the network in a single pass
- **num_epochs:** number of times the entire training dataset gets propagated through the network
- **steps_per_epoch:** number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size
- **validation_steps:** number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well
- **workers:** maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with

 ##### **Hyper Parameter Selection**
 ```
learning_rate = 0.001
batch_size = 64
num_epochs = 40
steps_per_epoch = 64
validation_steps = 50
workers = 70
 ```
 parameters should be explicitly stated with factual justifications
 Any choice of configurable parameters should also be explained
 how these values were obtained
 Epoch
Learning Rate
Batch Size
Etc.

##### **Training**
```
# Define the Keras model and compile it for training
model = models.Model(inputs=inputs, outputs=output_layer)
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

# Data iterators for loading the training and validation data
train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                               data_folder=os.path.join('..', 'data', 'train'),
                                               image_shape=image_shape,
                                               shift_aug=True)

val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                             data_folder=os.path.join('..', 'data', 'validation'),
                                             image_shape=image_shape)

logger_cb = plotting_tools.LoggerPlotter()
callbacks = [logger_cb]

model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)
```
Initial Training Curve. After a few Epochs:
![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch2.png)![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch4.png)

Towards the end, still improving slightly. Getting worse and oscillating
![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch13.png)![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch14.png)

### **Performance Results**
Here's training results from segmentation with the target:
![segmentation_withHero](/assets/segmentation_withHero.png)

Here's training results from segmentation with the target is not in the frame:
![segmentation_withoutHero](/assets/segmentation_withoutHero.png)
### **Complications and Limitations**
Computer graphics card, AWS, etc

### **Future Enhancements**
The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.
age, AngularJS powered HTML5 Markdown editor.
