import neuralbinding as nn
from utils import one_hot, normalize_image, init_weights, init_bias

# training function
def train(images, labels, epochs=5, lr=0.01):
    #NN dimensions
    in_dim = 784
    hidden_dim = 128
    out_dim = 10

    #param initialization L1
    w1 = init_weights(in_dim, hidden_dim)
    b1 = init_bias(hidden_dim)

    #param initialization L2
    w2 = init_weights(hidden_dim, out_dim)
    b2 = init_bias(out_dim)

    #loop over the epochs
    for epoch in range(epochs):
        total_loss = 0.0
        #updt weights after every img
        for x, y in zip(images, labels):
            #preprocessing
            x = normalize_image(x)
            t = one_hot(y, out_dim)

            #forward pass
            z1 = [0.0] * hidden_dim
            nn.dense_forward(x, w1, b1, z1, in_dim, hidden_dim)
            nn.relu_forward(z1, hidden_dim)

            y_pred = [0.0] * out_dim
            nn.dense_forward(z1, w2, b2, y_pred, hidden_dim, out_dim)

            #loss function
            loss = nn.mse_forward(y_pred, t, out_dim)
            total_loss += loss

            #backward pass
            grad_y = [0.0] * out_dim
            nn.mse_backward(y_pred, t, grad_y, out_dim)

            grad_z1 = [0.0] * hidden_dim
            grad_w2 = [0.0] * (out_dim * hidden_dim)
            grad_b2 = [0.0] * out_dim
            nn.dense_backward(z1, w2, grad_y, grad_z1, grad_w2, grad_b2, hidden_dim, out_dim)

            nn.relu_backward(z1, grad_z1, hidden_dim)

            grad_x = [0.0] * in_dim
            grad_w1 = [0.0] * (hidden_dim * in_dim)
            grad_b1 = [0.0] * hidden_dim
            nn.dense_backward(x, w1, grad_z1, grad_x, grad_w1, grad_b1, in_dim, hidden_dim)

            #SGD param update
            nn.sgd_update(w2, grad_w2, len(w2), lr)
            nn.sgd_update_bias(b2, grad_b2, len(b2), lr)

            nn.sgd_update(w1, grad_w1, len(w1), lr)
            nn.sgd_update_bias(b1, grad_b1, len(b1), lr)

        print("Epoch", epoch, "Loss", total_loss / len(images)) #prints avg loss