import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nonlinearity_utils import return_nonlinear_fn
import matplotlib.pyplot as plt

from utils import read_config, get_alphabet_x_y, get_regime_class
import numpy as np
import os
import matplotlib.animation as animation
from gaussian_capacity import get_gaussian_distribution


class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First dense layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))  # Second dense layer
        self.fc3 = nn.Linear(int(hidden_size / 2), output_size)  # Second dense layer
        self.fc4 = nn.Linear(hidden_size, output_size)  # Second dense layer
        self.softmax = nn.Softmax(dim=1)  # Softmax for probability output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        x = self.softmax(x)  # Apply softmax to get probabilities
        return x


# Training function
def train(model, criterion, optimizer, data_loader, num_epochs=10):
    opt_loss = []
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        opt_loss.append(loss.item())
    return opt_loss


def get_pdf_y_given_x(alphabet_x, alphabet_y, config):
    regime_class = get_regime_class(
        config, alphabet_x, alphabet_y, config["power1"], config["tanh_factor1"]
    )
    pdf_y_given_x_calculated = regime_class.get_pdf_y_given_x()

    return pdf_y_given_x_calculated, regime_class


def plot_comparison(
    alphabet_x,
    alphabet_y_comp,
    pdf_y_given_x_comp,
    alphabet_y,
    pdf_y_given_x,
    save_location,
    comp_str="NN",
):
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ind = 0
    line1 = ax.plot(
        alphabet_y_comp,
        pdf_y_given_x_comp[:, ind],
        label=comp_str,
    )
    line2 = ax.plot(
        alphabet_y,
        pdf_y_given_x[:, ind],
        label="Analytical",
    )
    ax.set_xlabel("y", fontsize=12)
    ax.set_ylabel("pdf(y|x)", fontsize=12)
    ax.legend(loc="best", fontsize=12)
    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )
    plt.minorticks_on()
    ax.grid(
        visible=True,
        which="minor",
        axis="both",
        color="gainsboro",
        linestyle=":",
        linewidth=0.5,
    )

    def update(ind):
        line1[0].set_ydata(pdf_y_given_x_comp[:, ind])
        line2[0].set_ydata(pdf_y_given_x[:, ind])
        ax.set_title("x=" + str(alphabet_x[ind]))
        return line1, line2

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=range(len(alphabet_x)), interval=300
    )

    ani.save(filename=save_location + "/Comp.gif", writer="pillow")


if __name__ == "__main__":

    criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for classification

    config = read_config(args_name="args_prob.yml")
    alphabet_x, alphabet_y, max_x, max_y = get_alphabet_x_y(
        config, config["power1"], config["tanh_factor1"]
    )
    y_boundaries = alphabet_y[1:] - (alphabet_y[1] - alphabet_y[0]) / 2
    # B = alphabet_y + (alphabet_y[1] - alphabet_y[0]) / 2
    # y_boundaries = torch.cat((A, B), 0)

    input_size = 1  # Number of input features- X is 1D vector
    hidden_size = config["hidden_size"]  # Number of neurons in the hidden layer

    output_size = len(y_boundaries) + 1  # Number of output classes for classification

    model = TwoLayerNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Creating noisy dataset
    torch.manual_seed(0)
    sigma_1, sigma_2 = 1, 1  # Noise standard deviations

    # X should be within [-max_x, max_x] while torch.rand is [0,1)

    x_train = torch.rand(config["training_size"], input_size) * 2 * max_x - max_x
    N1 = torch.randn_like(x_train) * sigma_1  # Input noise
    N2 = torch.randn(config["training_size"], 1) * sigma_2  # Output noise

    phi = return_nonlinear_fn(config=config, tanh_factor=config["tanh_factor1"])

    y_train = phi(x_train + N1).sum(dim=1, keepdim=True) + N2  # Applying phi(X+N1) + N2

    # Discretizing y_train into class labels
    y_train = torch.bucketize(y_train, boundaries=y_boundaries)

    dataset = TensorDataset(x_train, y_train.squeeze())
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Call training function
    opt_loss = train(
        model, criterion, optimizer, data_loader, num_epochs=config["num_epochs"]
    )

    print("Training complete!")
    save_location = (
        "NN/Sample="
        + str(config["min_samples"])
        + "/lr="
        + str(config["lr"])
        + "_batch="
        + str(config["batch_size"])
        + "_epochs="
        + str(config["num_epochs"])
        + "_training_size="
        + str(config["training_size"])
        + "_hidden_size="
        + str(hidden_size)
        + "/"
    )

    os.makedirs(save_location, exist_ok=True)

    plt.plot(opt_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(save_location + "opt_loss.png")

    # # Test the model

    outputs = model(alphabet_x.reshape(-1, 1))  # the shape is (X.length, Y.Length)

    pdf_y_given_x_comp = torch.transpose(outputs, 0, 1).detach().numpy()

    pdf_y_given_x, regime_class = get_pdf_y_given_x(alphabet_x, alphabet_y, config)

    plot_comparison(
        alphabet_x,
        alphabet_y,
        pdf_y_given_x_comp,
        alphabet_y,
        pdf_y_given_x,
        save_location=save_location,
        comp_str="NN",
    )

    pdf_x = get_gaussian_distribution(config["power1"], regime_class, config["complex"])
    mut_info_analytical = regime_class.new_capacity(
        pdf_x=pdf_x, pdf_y_given_x=pdf_y_given_x
    )
    mut_info_comp = regime_class.new_capacity(
        pdf_x=pdf_x, pdf_y_given_x=torch.tensor(pdf_y_given_x_comp).float()
    )
    print("Mutual information (analytical):", mut_info_analytical)
    print("Mutual information (NN):", mut_info_comp)
    print("Error in mutual information:", np.abs(mut_info_analytical - mut_info_comp))

    res = [
        ["Analytical", "NN", "Error"],
        [
            mut_info_analytical.detach().numpy(),
            mut_info_comp.detach().numpy(),
            np.abs(mut_info_analytical - mut_info_comp).detach().numpy(),
        ],
    ]
    np.savetxt(save_location + "results.csv", res, delimiter=",", fmt="%s")

    print("**** Results saved to", save_location)

    # # Save the model
    torch.save(model.state_dict(), save_location + "model.pth")

    # # Load the model
    # model = TwoLayerNN(input_size=2, hidden_size=10, output_size=2)
    # model.load_state_dict(torch.load("model.pth"))
    # model.eval()
    # # Test the loaded model
    # outputs = model(X)
