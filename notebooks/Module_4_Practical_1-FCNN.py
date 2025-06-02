import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Module 4: Practical 1 - Fully Connected Neural Networks""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Loss Functions
        #### Mean Square Loss
        $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

        #### Categorical Cross Entropy Loss
        $-\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$

        For the classification problems $y_{i,c}$ is equal to $1$ for the correct class, and $0$ otherwise, so we are really just averaging the logarithms of the probability the network outputs for the correct label in the training data.

          - Loss: $-\log(p_{\text{correct class}})$
          - Directly uses the index of the correct class
        """
    )
    return


@app.cell
def _(np):
    # Activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()
    return sigmoid, softmax


@app.cell(hide_code=True)
def _():
    # Create sample data for the network

    # Define input variables
    weather_examples = [
        [28.5, 45.0, 1015.2],
        [18.7, 78.3, 1008.5],
        [22.1, 85.6, 998.7]
    ]
    weather_categories = ["Sunny", "Rainy", "Stormy"]

    # Define input variables
    var_names = ["bias", "temperature (°C)", "humidity (%)", "pressure (hPa)"]
    return var_names, weather_categories, weather_examples


@app.cell
def _(mo):
    h1_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(4)])
    h2_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(4)])
    y1_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(2)])
    y2_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(2)])
    y3_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(2)])
    return (
        h1_weight_array,
        h2_weight_array,
        y1_weight_array,
        y2_weight_array,
        y3_weight_array,
    )


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(options={"Example 1":0, "Example 2":1, "Example 3":2}, value="Example 1")
    return (dropdown,)


@app.cell(hide_code=True)
def _(dropdown, mo, var_names, weather_categories, weather_examples):
    input_array = mo.ui.array([mo.ui.number(label=f'''{var_names[i]}''', value=weather_examples[dropdown.value][i-1]) for i in range(1,4)])
    label_dropdown = mo.ui.dropdown(options={weather_categories[0]:0, weather_categories[1]:1, weather_categories[2]:2}, value=weather_categories[dropdown.value])
    return input_array, label_dropdown


@app.cell(hide_code=True)
def _(
    dropdown,
    h_inputs,
    h_values,
    np,
    plt,
    sigmoid,
    v,
    var_values,
    w,
    weather_categories,
    y_inputs,
    y_values,
):
    import networkx as nx
    import math
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 3])

    # Main network diagram in top area (spanning 3 columns)
    ax_main = fig.add_subplot(gs[1, :])

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with positions
    pos = {
        "x1": (0, 0), "x2": (0, 0.5), "x3": (0, 1), "bias": (0, -0.5), # Input layer
        "h1": (1.5, 0.25), "h2": (1.5, 0.75),            # Hidden layer
        "y1": (3, 0), "y2": (3, 0.5), "y3": (3, 1)               # Output layer
    }

    # Add nodes
    G.add_nodes_from(["x1", "x2", "x3", "bias", "h1", "h2", "y1", "y2", "y3"])

    # Add edges
    edges = [
        ("x1", "h1"), ("x1", "h2"), 
        ("x2", "h1"), ("bias", "h1"), ("x2", "h2"),
        ("x3", "h1"), ("x3", "h2"),
        ("h1", "y1"), ("h2", "y1"),
        ("h1", "y2"), ("h2", "y2"),
        ("h1", "y3"), ("h2", "y3")
    ]
    G.add_edges_from(edges)

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_size=3000, 
            node_color=["lightblue", "lightblue", "lightblue", "lightblue", "lightgreen", "lightgreen", "salmon", "salmon", "salmon"],
            arrowsize=20, arrowstyle='->', width=1.5, ax=ax_main)

    # Add custom node labels with interpolated values
    node_labels = {
        "bias": "1",
        "x1": f"x₁\n{var_values[1]}", 
        "x2": f"x₂\n{var_values[2]}", 
        "x3": f"x₃\n{var_values[3]}", 
        "h1": f"\n\nh₁\n\n $\sigma({h_inputs[0]}) = {h_values[0]}$", 
        "h2": f"\n\nh₂\n\n $\sigma({h_inputs[1]}) = {h_values[1]}$", 
        "y1": f"\n\ny₁\n\n ${y_values[0]}$",
        "y2": f"\n\ny₂\n\n ${y_values[1]}$",
        "y3": f"\n\ny₃\n\n ${y_values[2]}$"
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=18, font_weight="bold", ax=ax_main)

    # Add edge labels with actual weight values
    edge_labels = {
        ("bias", "h1"): fr"$\times({w[0,0]})$", 
        ("x1", "h1"): fr"$\times({w[0,1]})$", 
        # ("x1", "h2"): fr"$\times({w[0,1]})$", 
        ("x2", "h1"): fr"$\times({w[0,2]})$", 
        # ("x2", "h2"): f"w3={w[1,1]}",
        ("x3", "h1"): fr"$\times({w[0,3]})$", 
        # ("x3", "h2"): f"w5={w[2,1]}",
        ("h1", "y1"): fr"$\times({v[0, 0]})$", 
        ("h2", "y1"): fr"$\times({v[0, 1]})$"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20, ax=ax_main)

    # Add layer titles
    ax_main.text(0, 1.5, "INPUT LAYER", fontsize=16, ha="center", fontweight="bold")
    ax_main.text(1.5, 1.5, "HIDDEN LAYER", fontsize=16, ha="center", fontweight="bold")
    ax_main.text(3, 1.5, "OUTPUT LAYER", fontsize=16, ha="center", fontweight="bold")

    ax_main.text(-0.5, 1, "Pressure", fontsize=16, ha="center", style='italic')
    ax_main.text(-0.5, 0.5, "Humidity", fontsize=16, ha="center", style='italic')
    ax_main.text(-0.5, 0, "Temperature", fontsize=16, ha="center", style='italic')


    # Add activation function labels
    ax_main.text(1.5, -0.3, "Activation: Sigmoid", fontsize=12, ha="center", style='italic')
    ax_main.text(3, -0.5, "Activation: Softmax", fontsize=12, ha="center", style='italic')

    example = dropdown.value
    ax_main.text(3.5, 1, f"{weather_categories[2]}", fontsize=16, ha="center", color = 'blue' if example == 2 else 'black')
    ax_main.text(3.5, 0.5, f"{weather_categories[1]}", fontsize=16, ha="center", color = 'blue' if example == 1 else 'black')
    ax_main.text(3.5, 0, f"{weather_categories[0]}", fontsize=16, ha="center", color = 'blue' if example == 0 else 'black')

    ax_main.text(2, -0.7, f"Loss: -log({y_values[example]}) = {-math.log(y_values[example])}", fontsize=20, ha="center", style='italic')


    # Color-code the layers with background shapes
    input_layer = plt.Rectangle((-0.5, -0.3), 0.8, 1.6, fill=True, alpha=0.1, color='blue')
    hidden_layer = plt.Rectangle((1, -0.05), 1, 1.1, fill=True, alpha=0.1, color='green')
    output_layer = plt.Rectangle((2.6, -0.3), 2, 1.5, fill=True, alpha=0.1, color='red')

    ax_main.add_patch(input_layer)
    ax_main.add_patch(hidden_layer)
    ax_main.add_patch(output_layer)

    # Add a sigmoid activation function plot in the bottom area
    ax_sigmoid = fig.add_subplot(gs[0, 1:2])
    x = np.linspace(-6, 6, 100)
    y = sigmoid(x)

    # Plot the sigmoid function
    ax_sigmoid.plot(x, y, 'b-', linewidth=2)
    ax_sigmoid.set_title('Sigmoid Activation Function: σ(z) = 1/(1+e^(-z))', fontsize=14)
    ax_sigmoid.set_xlabel('Input (z)', fontsize=12)
    ax_sigmoid.set_ylabel('Output: σ(z)', fontsize=12)
    ax_sigmoid.grid(True, alpha=0.3)
    ax_sigmoid.set_xlim(-6, 6)
    ax_sigmoid.set_ylim(-0.1, 1.1)


    ax_main.text(3, 1.4, f"softmax({y_inputs[2], y_inputs[1], y_inputs[1]})", fontsize=18, ha="center")

    plt.tight_layout()
    plt.show()
    return (
        G,
        GridSpec,
        ax_main,
        ax_sigmoid,
        edge_labels,
        edges,
        example,
        fig,
        gs,
        hidden_layer,
        input_layer,
        math,
        node_labels,
        nx,
        output_layer,
        pos,
        x,
        y,
    )


@app.cell(hide_code=True)
def _(
    dropdown,
    h1_weight_array,
    h2_weight_array,
    input_array,
    label_dropdown,
    mo,
    np,
    sigmoid,
    softmax,
    y1_weight_array,
    y2_weight_array,
    y3_weight_array,
):
    var_values = [1, input_array[0].value, input_array[1].value, input_array[2].value]
    #[1, example["temp"], example["humidity"], example["pressure"]]

    # Define weights (randomly initialized)
    w = np.round(np.random.uniform(-1, 1, size=(2, 4)), 2)  # 3 inputs × 2 hidden neurons
    # w = np.array([
    #     [ 0.3, -0.2,  0.1,  0.4],  # temperature weights
    #     [-0.1,  0.5,  0.2, -0.3],  # humidity weights
    #     [ 0.2,  0.1, -0.4,  0.1]   # pressure weights
    # ])

    v = np.round(np.random.uniform(-1, 1, size=(3, 4)), 2)       # 2 hidden neurons × 1 output
    # Weights for output layer (4 hidden neurons × 3 output classes)
    # v = np.array([
    #     [ 0.5,  0.1, -0.3],  # h1 to outputs
    #     [-0.2,  0.4,  0.1],  # h2 to outputs
    #     [ 0.1, -0.1,  0.5],  # h3 to outputs
    #     [-0.1,  0.3,  0.2]   # h4 to outputs
    # ])


    w[0,0] = h2_weight_array[3].value
    w[0,1] = h2_weight_array[2].value
    w[0,2] = h2_weight_array[1].value
    w[0,3] = h2_weight_array[0].value

    w[1,0] = h1_weight_array[3].value
    w[1,1] = h1_weight_array[2].value
    w[1,2] = h1_weight_array[1].value
    w[1,3] = h1_weight_array[0].value

    v[0,0] = y1_weight_array[0].value
    v[0,1] = y1_weight_array[1].value
    v[1,0] = y2_weight_array[0].value
    v[1,1] = y2_weight_array[1].value
    v[2,0] = y3_weight_array[0].value
    v[2,1] = y3_weight_array[1].value


    # Calculate hidden layer values
    h_inputs = np.zeros(2)
    for i in range(2):
        h_inputs[i] = np.round(np.sum([w[i, j] * var_values[j] for j in range(4)]), 2)
    h_values = np.round(sigmoid(h_inputs), 2)

    # Calculate output values
    y_inputs = np.zeros(3)
    for i in range(3):
        y_inputs[i] = np.round(np.sum([v[i, j] * h_values[j] for j in range(2)]), 2)
    y_values = np.round(softmax(y_inputs), 2) # Using softmax for multi-class classification

    mo.vstack([dropdown, mo.hstack([input_array[::-1], label_dropdown], justify="start"), mo.hstack([h1_weight_array, h2_weight_array, mo.vstack([y3_weight_array, y2_weight_array, y1_weight_array])])], justify="start")
    return h_inputs, h_values, i, v, var_values, w, y_inputs, y_values


@app.cell
def _(np):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    BATCH_SIZE = 32
    CLASSES = np.array(
        [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    )

    # Prepare the Data
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return (
        BATCH_SIZE,
        CLASSES,
        DataLoader,
        nn,
        optim,
        test_dataset,
        torch,
        torchvision,
        train_dataset,
        transform,
        transforms,
    )


@app.cell
def _(train_dataset):
    # check the shape of the first image
    train_dataset[0][0].shape
    return


@app.cell
def _(CLASSES, plt, train_dataset):
    _img, _label = train_dataset[0]
    print("Label: ", _label)
    print("Class: ", CLASSES[_label])
    plt.imshow(_img.permute(1,2,0).squeeze())
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that each training example is a tuple containing the three dimensional image tensor (C by H by W) and the label.""")
    return


@app.cell
def _(train_dataset):
    print("Some individual pixel: ", train_dataset[54][0][1, 12, 13])
    print("Corresponding Label: ", train_dataset[54][1])
    return


@app.cell
def _(CLASSES, np, plt, train_dataset):
    _random_index = np.random.randint(len(train_dataset))
    _img, _label = train_dataset[_random_index]
    print("Label: ", _label)
    print("Class: ", CLASSES[_label])
    plt.imshow(_img.permute(1,2,0).squeeze())
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""PyTorch has a special *DataLoader* class that takes care of some of the tedious details of constructing batches from the dataset.""")
    return


@app.cell
def _(BATCH_SIZE, DataLoader, test_dataset, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, train_loader


@app.cell
def _(train_dataset, train_loader):
    print("Length of the train dataset: ", len(train_dataset))
    print("Length of list(train_loader): ", len(list(train_loader)))
    print("Type of the first element of list(train_loader): ", type(list(train_loader)[0]))
    print("Type of the first element of list(train_loader)[0]: ", type(list(train_loader)[0][0]))
    print("Shape of the first element of list(train_loader)[0]: ", list(train_loader)[0][0].shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Dataloaders** provide multiple ways to access the data, either by converting it into a **Python list** or by using an **iterable**.  

        Using `list(train_loader)`, as we have, loads the **entire dataset into memory**, which can be **slow** and even **fail** when dealing with large datasets.  

        Since **neural network training algorithms process data in batches**, it is more efficient to use an **iterator**. Instead of retrieving the first batch like this:  
        ```python
        list(train_loader)[0]
        ```
        which loads everything into memory, we use:
        ```python
        next(iter(train_loader))
        ```
        This approach retrieves only the first batch without loading the entire dataset, making it memory-efficient and faster.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Let's load the first batch of our data (image and label) and display it using the `matplotlib` library.

        Recall that the shape returned by 
        ```python
        next(iter(train_loader))
        ```
        is 32 by 3 by 32 by 32. This shape represents the batch size, number of channels, height, and width of the image, respectively.
        """
    )
    return


@app.cell
def _(CLASSES, plt, train_loader):
    next_batch_images, next_batch_labels = next(iter(train_loader))
    _first_img = next_batch_images[0] # retrieve the first image from the batch of 32
    _first_label = next_batch_labels[0] # retrieve the first label from the batch of 32
    plt.imshow(_first_img.permute(1, 2, 0)) # imshow requires the image to be in height x width x channels format
    plt.show()
    print("Label: ", CLASSES[_first_label])
    return next_batch_images, next_batch_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Why is the first image different from when we used the dataset directly?""")
    return


@app.cell
def _():
    # Parameters
    NUM_CLASSES = 10
    EPOCHS = 10
    return EPOCHS, NUM_CLASSES


@app.cell
def _(NUM_CLASSES, nn, torch):
    # Build the model
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 32 * 3, 200)
            self.fc2 = nn.Linear(200, 150)
            self.fc3 = nn.Linear(150, NUM_CLASSES)

        def forward(self, x):
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.softmax(self.fc3(x), dim=1)
            return x

    model = MLP()
    print(model)

    # Compare to TensorFlow
    # input_layer = layers.Input(shape=(32, 32, 3))
    # x = layers.Flatten()(input_layer)
    # x = layers.Dense(units=200, activation = 'relu')(x)
    # x = layers.Dense(units=150, activation = 'relu')(x)
    # output_layer = layers.Dense(units=10, activation = 'softmax')(x)
    # model = models.Model(input_layer, output_layer)
    return MLP, model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Training the model""")
    return


@app.cell
def _(EPOCHS, model, nn, optim, torch, train_loader):
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    correct = 0
    total = 0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for ind, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if ind % 100 == 99:
                print(f"[{epoch + 1}, {ind + 1}] accuracy: {correct/total:.3f}, loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")
    return (
        correct,
        criterion,
        data,
        epoch,
        ind,
        inputs,
        labels,
        loss,
        optimizer,
        outputs,
        predicted,
        running_loss,
        total,
    )


@app.cell
def _(correct, data, model, test_loader, torch, total):
    # Evaluation
    _correct = 0
    _total = 0
    with torch.no_grad():
        for _data in test_loader:
            _images, _labels = data
            _outputs = model(_images)
            _, _predicted = torch.max(_outputs.data, 1)
            _total += _labels.size(0)
            _correct += (_predicted == _labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f}%")
    return (accuracy,)


@app.cell(hide_code=True)
def _(accuracy, mo):
    mo.md(
        fr"""
        The model has an **accuracy of {accuracy:.2f}%** on the test set, which is **better than random guessing** (10 classes).  

        However, this accuracy is **low** compared to **state-of-the-art models**.  

        The **simple model** we built has **limited capacity** to learn the **complex patterns** in the CIFAR-10 dataset.  

        Next, we will build a **more advanced model** using convolutional neural network (CNN) to **improve accuracy** and **learn more complex patterns** in the data.
        """
    )
    return


@app.cell
def _(CLASSES, model, np, plt, test_loader, torch):
    _images, _labels = next(iter(test_loader))
    _outputs = model(_images)
    _, preds = torch.max(_outputs, 1)
    preds_single = CLASSES[preds.numpy()]
    actual_single = CLASSES[_labels.numpy()]
    n_to_show = 10
    indices = np.random.choice(range(len(_images)), n_to_show)

    _fig = plt.figure(figsize=(15, 3))
    _fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for _i, idx in enumerate(indices):
        img = _images[idx].numpy().transpose((1, 2, 0))
        ax = _fig.add_subplot(1, n_to_show, _i + 1)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            "pred = " + str(preds_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.7,
            "act = " + str(actual_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.imshow(img)
    plt.show()
    return actual_single, ax, idx, img, indices, n_to_show, preds, preds_single


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
