 ### **Project Summary**
 

### **Network Architecture**
 all network architecture should be explained, 
 layer of the network architecture and the role that it plays in the overall network
 benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data
 ![Network_Arch_FCN](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/Network_Arch_FCN.PNG)
 provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.
 
 ##### **Bilinear Upsampling**
 Upsample:
 ```
 def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
 ```
 ##### **Encoder Block**
 The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.
 ```
 def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```
 ##### **Decoder**
 Decoder
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
### **FCN Model**
FCNs
![FCN_Model](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/FCN_Model.PNG)
##### **Encoder Layers**
Encoder
```
def fcn_model(inputs, num_classes):
    # Encoder Blocks. 
    layer_1 = encoder_block(inputs, 64, 2)
    layer_2 = encoder_block(layer_1, 128, 2)
```
 ##### **1 x 1 Convolution Layer**
The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.
The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.
```
    # 1x1 Convolution layer.
    layer_3 = conv2d_batchnorm(layer_2, 256, kernel_size=1, strides=1)
```
##### **Decoder Blocks**
Decoder
```
    layer_4 = decoder_block(layer_3, layer_1, 128)
    layer_5 = decoder_block(layer_4, inputs, 64)
```
##### **Output Layer**
Output
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
Initial Training Curve
![TrainingCurve_Epoch2](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/TrainingCurve_Epoch2.png)
After a few Epochs
![TrainingCurve_Epoch2](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/TrainingCurve_Epoch4.png)
Towards the end, still improving slightly
![TrainingCurve_Epoch2](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/TrainingCurve_Epoch13.png)
Getting worse and oscillating
![TrainingCurve_Epoch2](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/TrainingCurve_Epoch14.png)

### **Performance Results**
Here's training results from segmentation with the target:
![segmentation_withHero](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/segmentation_withHero.PNG)

Here's training results from segmentation with the target is not in the frame:
![segmentation_withoutHero](https://github.com/mattlubbers/Udacity-RoboND-DeepLearning/assets/segmentation_withoutHero.PNG)
### **Complications and Limitations**
Computer graphics card, AWS, etc

### **Future Enhancements**
The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.
age, AngularJS powered HTML5 Markdown editor.
