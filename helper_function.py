###############
# Function to create loss and accuracy curve of training v/s validation 
import matplotlib.pyplot as plt
def plot_loss_curve(history):

    '''
    Plots the loss and accuracy curve of the model 

    '''
    loss = history.history['loss']
    accuracy = history.history['accuracy']

    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot the losses
    plt.plot(epochs,loss, label='Training Loss')
    plt.plot(epochs,val_loss, label='Validation Loss')
    plt.title("Training v/s Validation Loss")
    plt.legend()

    # Plot the accuracy
    plt.figure()
    plt.plot(epochs,accuracy, label='Training Accuracy')
    plt.plot(epochs,val_accuracy, label='Validation Accuracy')
    plt.title("Training v/s Validation Accuracy")
    plt.legend()


###############################
# Create a functionn to import an image and resize it to be able to use by our model
import tensorflow  as tf
def load_prep_image(filename, img_shape = 224 or 256):
    """
    Reads an image from filename turn it into a tensor and reshapes it to 
    (img_shape, img_shape, color_channel)

    """

    # Read in the image
    img = tf.io.read_file(filename)

    # Decode the read file into tensor
    img = tf.image.decode_image(img) 

    # Resize the image
    img = tf.image.resize(img, size = [img_shape, img_shape])

    # Rescale the image and get all values b/w 0 and 1
    img = img / 255.
    return img
#############################

# Function to visualize prediction
def visualize_predictions(model, data):
    images, labels = next(data)
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int)  # Apply threshold

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        true_label = "Cat" if labels[i] == 0 else "Dog"
        pred_label = "Cat" if predictions[i] == 0 else "Dog"
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis("off")
    plt.show()

########################################

# Function to make prediction on custmom image
def pred_and_plot(model, filename, class_names):

    '''
    Imports an image located at filename, makes a prediction with the model and plots the image with 
    the predicted class as the title
    
    '''

    # Import the target image
    img = load_prep_image(filename)

    # Make a prediction 
    pred = model.predict(tf.expand_dims(img, axis = 0))
    print(pred)
    # Get the predictied class
    pred_class = class_names[int(tf.round(pred))]
    print(pred_class)
    # Plot the image and predicted class
    plt.figure()
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)

########################################

# Predict on test data
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
def evaluation_metrics(model, test_data, class_names):
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class indices

    # True labels (assuming test_data has labels in the second element)
    y_true = test_data.classes
    y_labels = [class_names[label] for label in y_true]

    # Calculate accuracy
    accuracy = accuracy_score(y_true, predicted_labels)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, predicted_labels))