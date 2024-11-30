import tensorflow as tf      
from tensorflow.keras import layers, models  
import matplotlib.pyplot as plt              


# Load the CIFAR-10 dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Normalize pixel values to range [0, 1] 
x_train = x_train.astype('float32') / 255.0 
x_test = x_test.astype('float32') / 255.0 
# Convert class vectors to one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, 10) 
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

# Display a few images with their Labels 
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

fig, axes = plt.subplots (1, 10, figsize=(15, 5)) 
for i in range(10): 
    axes[i].imshow(x_train[i]) 
    axes[i].axis('off') 
    axes[i].set_title(class_names[y_train[i].argmax()]) 
plt.show()

def build_model():
    model = models.Sequential() 
    #Convolutional Block 1 
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(32, 32, 3))) 
    model.add(layers. MaxPooling2D((2, 2))) 
    
    #Convolutional Block 2 
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) 
    model.add(layers.MaxPooling2D((2, 2))) 

    #Convolutional Block 3 
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) 

    #Fully Connected Layers 
    model.add(layers.Flatten()) 
    model.add(layers.Dense(64, activation = 'relu')) 
    model.add(layers.Dense(10, activation='softmax')) # Output Layer for 10 classes

    return model 

# Build the model 
cnn_model = build_model() 
cnn_model.summary()

cnn_model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy']) 

# Train the model 
history = cnn_model.fit(x_train, y_train, 
                        epochs=10, batch_size=64, 
                        validation_data=(x_test, y_test)) 
#Evaluate on test data 
test_loss, test_acc = cnn_model.evaluate(x_test, y_test, verbose=2) 
print(f"Test Accuracy: {test_acc:.2f}")


#Plot accuracy and Loss 
plt.figure(figsize=(12, 5)) 

#Accuracy plot 
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 


# Loss plot 
plt.subplot(1, 2, 2) 
plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.title('Loss Over Epochs') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 

plt.show()

#Make predictions 
predictions = cnn_model.predict(x_test) 
# Display some predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 5)) 
for i in range(5): 
    axes[i].imshow(x_test[i]) 
    axes[1].axis('off') 
    pred_label = class_names[predictions[i].argmax()] 
    true_label = class_names[y_test[i].argmax()] 
    axes[i].set_title(f"True: (true_label)\nPred: (pred_label)", color="green" if pred_label == true_label else "red") 
plt.show()


