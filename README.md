# Cat-vs.-Dog-Image-Classification

This project implements a binary image classification model to distinguish between cats and dogs using TensorFlow/Keras.

## Dataset
The dataset used in this project is the **Cats and Dogs Images** dataset from Kaggle. It contains images of cats and dogs, organized into training, validation, and test sets.

- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/your-dataset-link)
  
Although it didn't havd `val/` folder, So I created one.
The dataset is organized into three directories:
- `train/`: Training images for cats and dogs.
- `val/`: Validation images for cats and dogs.
- `test/`: Test images for cats and dogs.

## Model Architectures
1. **Model 0**: A simple CNN with two convolutional layers.
2. **Model 1**: A CNN with data augmentation.
3. **Model 2**: Transfer learning using the Xception model.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

Install the dependencies using:
```bash
pip install -r requirements.txt
