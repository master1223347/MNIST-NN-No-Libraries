'''
these are python standard imports, technically the project can be done without these but its just a lot more messy,
these are here purely for reproducibility
'''
import struct

def load_mnist_images(file_path):
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError("Invalid MNIST image file magic number")

        raw = f.read()

        images = []
        image_size = rows * cols

        for i in range(num):
            start = i * image_size
            end = start + image_size

            # normalize to [0, 1]
            img = [raw[j] / 255.0 for j in range(start, end)]
            images.append(img)

    return images


def load_mnist_labels(file_path):
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError("Invalid MNIST label file magic number")

        labels = list(f.read())

        if len(labels) != num:
            raise ValueError("Label count mismatch")

    return labels


def load_mnist(image_path, label_path):
    images = load_mnist_images(image_path)
    labels = load_mnist_labels(label_path)

    if len(images) != len(labels):
        raise ValueError("Image and label count mismatch")

    return images, labels