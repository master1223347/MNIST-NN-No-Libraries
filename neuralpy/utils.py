import random #used only for initialization 

# One-hot encoding
def one_hot(label, num_classes):
    vec = [0.0] * num_classes
    vec[label] = 1.0
    return vec

# input normalization
def normalize_image(image):
    out = []
    for p in image:
        out.append(p / 255.0)
    return out

#weight initialization 
def init_weights(in_dim, out_dim):
    weights = []
    scale = (1.0 / in_dim) ** 0.5  # simple variance control

    for i in range(out_dim):
        for j in range(in_dim):
            weights.append(random.uniform(-scale, scale))

    return weights

# bias initialization
def init_bias(out_dim):
    b = []
    for i in range(out_dim):
        b.append(0.0)
    return b

#well, shuffles data
def shuffle_data(images, labels):
    if len(images) != len(labels):
        raise ValueError("Image and label count mismatch")

    indices = list(range(len(images)))
    random.shuffle(indices)

    shuffled_images = [images[i] for i in indices]
    shuffled_labels = [labels[i] for i in indices]

    return shuffled_images, shuffled_labels

# calculates accuracy
def accuracy(predictions, labels):
    correct = 0

    for y_pred, y_true in zip(predictions, labels):
        max_idx = 0
        max_val = y_pred[0]

        for i in range(1, len(y_pred)):
            if y_pred[i] > max_val:
                max_val = y_pred[i]
                max_idx = i

        if max_idx == y_true:
            correct += 1

    return correct / len(labels)