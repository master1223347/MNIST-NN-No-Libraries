'''
these are python standard imports, technically the project can be done without these but the user will need to manually decompress
files, these are here purely for reproducibility
'''
import struct

def load_mnist_images(file_path):
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number")
        # read the rest as unsigned bytes
        images = f.read()
        images = [pixels / 255.0 for pixels in images]  # normalize to [0,1]
        # reshape to list of lists
        images = [images[i * rows * cols:(i + 1) * rows * cols] for i in range(num)]
    return images

def load_mnist_labels(file_path):
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number in MNIST label file!")
        labels = list(f.read())
    return labels
