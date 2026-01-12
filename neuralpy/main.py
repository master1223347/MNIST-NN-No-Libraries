from data import load_mnist
from utils import shuffle_data, accuracy
from train import train
import neuralbinding as nn

def main():
    # Paths to MNIST files
    train_images_path = "data/train-images.idx3-ubyte"
    train_labels_path = "data/train-labels.idx1-ubyte"
    test_images_path = "data/t10k-images.idx3-ubyte"
    test_labels_path = "data/t10k-labels.idx1-ubyte"

    # Load MNIST data
    train_images, train_labels = load_mnist(train_images_path, train_labels_path)
    test_images, test_labels = load_mnist(test_images_path, test_labels_path)

    # Shuffle training data
    train_images, train_labels = shuffle_data(train_images, train_labels)

    # Train the network and get trained weights
    w1, b1, w2, b2 = train(train_images, train_labels, epochs=5, lr=0.01)

    # Evaluate on the test set
    hidden_dim = 128
    out_dim = 10

    predictions = []
    for x in test_images:
        z1 = [0.0] * hidden_dim
        nn.dense_forward(x, w1, b1, z1, 784, hidden_dim)
        nn.relu_forward(z1, hidden_dim)

        y_pred = [0.0] * out_dim
        nn.dense_forward(z1, w2, b2, y_pred, hidden_dim, out_dim)

        predictions.append(y_pred)

    acc = accuracy(predictions, test_labels)
    print("Test Accuracy:", acc)

if __name__ == "__main__":
    main()