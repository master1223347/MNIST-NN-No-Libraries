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

