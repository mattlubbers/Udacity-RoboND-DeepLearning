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

##### **Stride = 2**
A stride of 2 moves the filter 2 pixels at a time, which produces small spatial volumes with a nearly negligible loss of quality

##### **Zero Padding = 1**
A minimal border of zeros surrounding the image prevents losing image information, as well as controls the spatial size of the output volume

##### **Layer Height and Width Calculation**
To determine the convolution layer height and width we use the equation: 
**(W − F + 2 * P) / S + 1**

Therefore the height and width of our first layer will be **80x80x64**: 
**(160 − 3 + 2 * 1) / 2 + 1**

We now have our Network Architecture defined, and will begin to construct our 5 layer FCN, with the 1x1 convolution layer in the middle. It will look something like this:

 ![Network_Arch_FCN](/assets/Network_Arch_FCN.PNG)
 
 ##### **Skip Connections**
 Something that has proven results for reducing training loss is Skip Connections. When the input layers are swept for enconding feature vectors, an common occurence is that they will capture the granular details, but miss the larger perspective picture. Skip Connections help mitigate this by bypassing adjacent layers for reference. 
 
 ### **FCN Model**
Now that we have properly defined the Architecture and sizing of our FCN, so the next step is to start to implement it! This will be composed of the input image, 2 Encoding layers, a 1x1 convolutional layer, 2 Decoding layers, and finally the ouput. It is important to maintain symmetry by including the same number and sizing of the Encoding and Decoding layers. The FCN Model can be seen here: 

![FCN_Model](/assets/FCN_Model.PNG)

  ##### **Encoder Block**
The objective of the Encoder is to sweep the input image, and extract features from the image. For our Encoder, we are using a depthwise separable 2D convolution. Separable convolutions act on each input channel separately, followed by a pointwise convolution that merges the resulting output channels. It's a method to factorize a convolution kernel into two smaller kernels, and therefore improve efficiency. More information can be found [here](https://keras.io/layers/convolutional/) in the Keras documentation.

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
Now that the Encoder functions are defined, implementing the layers is as simple as calling our encoder block function for each corresponding layer. This will take the original image input, and create the Encoder layers prior to our 1x1 convolution layer:
```
def fcn_model(inputs, num_classes):
    # Encoder Blocks. 
    layer_1 = encoder_block(inputs, 64, 2)
    layer_2 = encoder_block(layer_1, 128, 2)
```
 ##### **1x1 Convolution Layer**
As mentioned previously in the Network Architecture, a 1x1 convolution layer has many benefits including:
- Both capable of classification, and preservation of spatial information in the image. (Is this a car, and where in the image is this car?)
- It allows flexibility of different sized input images 
- More layers can be added to create depth in the network, without causing massive computational cost

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
Similar to the Encoding layers, but in opposite direction, we will begin to Decode our layers to gradually transition from the 1x1x256 to the desired output image size of 160x160x3. A critical element of the Decoding process is the Bilinear Upsampling, which helps reconstruct our downsampled image.

Bilinear Upsampling helps us transform the downsampled image into the original input image dimensions. This can be represented best by this diagram below:

![Bilinear_Upsampling](/assets/Bilinear_Upsampling.png)

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
With our FCN implemented, we now need to set the hyper parameters. First let's define the parameters we will be using: 
 ##### **Hyper Parameter Definition**
- **learning_rate:** A value that multiplies the **derivative of the loss function** prior to subtracting from the corresponding **weight**
- **batch_size:** Number of **images** that are propagated in a **single cycle**
- **num_epochs:** Number of **cycles** that the entire training set propagates through the **network**
- **steps_per_epoch:** Number of **batches** that propagate through the network in a single **epoch**
- **validation_steps:** Number of **batches** for **validation images** that propagate through the network in a single **epoch**
- **workers:** Number of **compute** processes **allocated**

 ##### **Hyper Parameter Selection**
 
- **learning_rate:** The initial value I chose was a magnitude of 10 higher. With a learning rate of 0.01, the model trained very quickly, however the end loss was too high, and therefore resulted in an overall IoU accuracy score of less than the target 40%. By slowing down the learning rate to 0.001, the convergence time took a bit longer, but the training loss was significantly improved.
- **batch_size:** It's recommended that the selected batch size is equivalent to the filter size selected for the initial convolutional layer. I therefore set this parameter to 64, and the results were sufficient.
- **num_epochs:** Initially, I set the number of epochs to 40 (along with learning_rate of 0.01) and allowed the training to complete for the entire number of epochs. After lowering the learning_rate to 0.001 on my second attempt, I was closely monitoring the convergence as well as the training loss for the model. After the 8th epoch, both the training and validation loss began to oscillate. It's quite common in training to suspend the training prior to reaching the target number of epochs if the training loss worsens. For this reason, I monitored the training loss and suspended the training during the 15th epoch when the training loss had reached a value of 0.0195.
- **steps_per_epoch:** The steps per epoch should be calculated based on the number of training images and the **batch_size**. Since we have 4,131 images in our training set, this would equate to the number of images divided by the batch size: **4131 / 64**
- **validation_steps:** This parameter was unchanged, however it resulted in acceptable validation loss and enabled meeting the targeted 40% IoU score.
- **workers:** The initial training was completed on my personal computer without a powerful GPU. For the second training attempt, I used the AWS elastic compute, and therefore increased the number of allocated compute processes.

 ```
learning_rate = 0.001
batch_size = 64
num_epochs = 40
steps_per_epoch = 64
validation_steps = 50
workers = 70
 ```

##### **Training**
This is the supporting code provided by the Udacity project to perform the training function:
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
Both the training and validation loss curves were relatively linear with a steady rate of decline initially. After the 4th epoch of training, the training loss continued with a steady decline to reach 0.0252, yet the validation loss curve began to increase. The training and validation loss curves after the 2nd and 4th epoch can be referenced below:

![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch2.png)![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch4.png)

Initially the number of epochs was set to 50, however I suspended the training due to the oscillations in the training and validation loss curves. After the fourteenth epoch, the training loss had reached 0.0195 and the validation loss again began to rise. The training and validation loss curves after the thirteenth and fourteenth epoch can be referenced below:  

![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch13.png)![TrainingCurve_Epoch2](/assets/TrainingCurve_Epoch14.png)

##### Model Weights
After the model has been trained on AWS, the model weights were written to a file in HDF format:
```
weight_file_name = 'model_weights'
model_tools.save_network(model, weight_file_name)
```
### **Performance Results**
The result from the trained model provides the semantic segmentation for images that include our target, as well as images without the target, but contain other environment features and pedestrians. 

##### Semantic Segmentation with the target:
![segmentation_withHero](/assets/segmentation_withHero.png)

##### Semantic Segmentation without the target:
![segmentation_withoutHero](/assets/segmentation_withoutHero.png)

### Results!
Quad is following behind the target:
```
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.99420453948768
average intersection over union for other people is 0.29249103914013497
average intersection over union for the hero is 0.8928129750460355
number true positives: 539, number false positives: 0, number false negatives: 0
```
Quad is on patrol and the target is not visable:
```
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.9808836391767214
average intersection over union for other people is 0.6079952255542386
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 86, number false negatives: 0
```
Detection from far away:
```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.99564611955333
average intersection over union for other people is 0.39380282310420306
average intersection over union for the hero is 0.24797525731904344
number true positives: 155, number false positives: 4, number false negatives: 146
```
True Positive weighted score:
```
0.7462365591397849
```
The IoU without the target:
```
0.570394116183
```
Final IoU score:
```
0.425648942614
```

### **Complications and Limitations**
The project was very informative and required external research outside the classroom to understand the reasons behind the design of the Network Architecture, Sizing, and Hyper Parameter tuning. Additional resources for these activities would have been helpful, however in the process of searching, the learning process continues!

From the hardware and model training perspective, my personal computer had severe limitations due to the insufficient graphical computation power required for this project. The initial training was attempted on this machine, and it took over 8 hours, as opposed to AWS completing this same task in nearly 2 hours.

Setting up AWS elastic compute for the first time was a helpful learning experience, but also took time to understand the necessary steps for implementation, as well as preventing unwanted charges. Now that I'm familiar with the process, it's relatively straight forward, and I look forward to using this service for future projects.

### **Future Enhancements**
There are many enhancements that would improve the end resulting score in this project:
- More training data
- Reviewed problem areas that resulted in false positives, and collect data in these specific use-cases
- More allocated validation data
- Refined validation step tuning - Resulted in poor validation loss
- Additional Hyper Parameter tuning
- Variable Learning Rate - It's common to vary the learning rate throughout the epoch progression, and this would have helped lower the training loss and resulted in a higher IoU score

Furthermore, this FCN model could be applied to other agents besides pedestrians, including: animals, vehicles, or buildings. However, this transition to alternative agent classification would require specific training and validation data, as well as additional hyper parameter tuning to achieve similar results.

