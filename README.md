# Fashion MNIST CNN Classifier

A Convolutional Neural Network (CNN) model built with TensorFlow/Keras to classify images in the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The dataset consists of 10 classes of grayscale images, each representing different types of clothing items.

## Features
- Preprocessing: Normalization and reshaping of images for CNN compatibility.
- CNN Architecture:
  - **3 Conv2D Layers** with ReLU activation and MaxPooling2D for feature extraction.
  - **Dense Layers** for classification with softmax activation.
- Metrics: Training and validation accuracy and loss tracking.
- Visualization: Accuracy and loss plots for performance analysis.

---

## Dataset
The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) consists of:
- **Training Data**: 60,000 28x28 grayscale images.
- **Test Data**: 10,000 28x28 grayscale images.
- **Classes**:
  - T-shirt/Top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle Boot

---

## Installation

### Prerequisites
Ensure you have Python 3.7+ installed. Then install the required libraries:
```bash
pip install tensorflow matplotlib
```

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Real-J/fashion-mnist-cnn.git
   cd fashion-mnist-cnn
   ```

2. Run the Python script:
   ```bash
   python fashion_mnist_cnn.py
   ```

3. View the accuracy and loss plots generated after training.

---

## Model Architecture
The CNN model consists of:
1. **Convolutional Layers**:
   - Extract meaningful features from input images using filters.
   - ReLU activation function for non-linearity.
2. **MaxPooling Layers**:
   - Down-sample the feature maps to reduce dimensionality.
3. **Dense Layers**:
   - Fully connected layers for classification.
   - Output layer uses softmax for multi-class probabilities.

### Summary:
```plaintext
Layer (type)               Output Shape           Param #
================================================================
conv2d (Conv2D)            (None, 26, 26, 32)     320
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)   0
conv2d_1 (Conv2D)          (None, 11, 11, 64)     18496
max_pooling2d_1 (MaxPooling2D) (None, 5, 5, 64)   0
conv2d_2 (Conv2D)          (None, 3, 3, 64)       36928
flatten (Flatten)          (None, 576)            0
dense (Dense)              (None, 64)             36928
dense_1 (Dense)            (None, 10)             650
================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

---

## Results
After 10 epochs of training:
- **Test Accuracy**: ~X.XX (update this after running the model)
- **Test Loss**: ~X.XX (update this after running the model)



---

## Potential Improvements
- **Data Augmentation**: Introduce transformations like rotation, flipping, and zooming for better generalization.
- **Regularization**: Use Dropout and L2 regularization to combat overfitting.
- **Transfer Learning**: Experiment with pre-trained models like MobileNetV2 for enhanced performance.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

