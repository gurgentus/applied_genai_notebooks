import marimo

__generated_with = "0.11.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Module 4: Practical 3 - Convolutional Neural Networks""")
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    BATCH_SIZE = 32

    # Prepare the Data
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return (
        BATCH_SIZE,
        DataLoader,
        F,
        nn,
        np,
        optim,
        plt,
        test_dataset,
        torch,
        torchvision,
        train_dataset,
        transform,
        transforms,
    )


@app.cell
def _(train_dataset):
    train_dataset[0][0].shape
    return


@app.cell
def _(plt, train_dataset):
    first_img, first_label = train_dataset[0]
    print("Label: ", first_label)
    plt.imshow(first_img.permute(1,2,0).squeeze())
    plt.show()
    return first_img, first_label


@app.cell
def _(first_label, np):
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
    CLASSES[first_label]
    return (CLASSES,)


@app.cell
def _(train_dataset):
    print("Some individual pixel: ", train_dataset[54][0][1, 12, 13])
    print("Corresponding Label: ", train_dataset[54][1])
    return


@app.cell
def _(CLASSES, np, plt, train_dataset):
    _random_index = np.random.randint(len(train_dataset))
    _img, _label = train_dataset[_random_index]
    print("Label: ", _label, CLASSES[_label])
    plt.imshow(_img.permute(1,2,0).squeeze())
    plt.show()
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


@app.cell
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
def _(F, nn):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # without padding output image size = (W-F)/S+1
            # output tensor dimensions: (?, 16, 32, 32)
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input channels = 3, Output channels = 16
            # output tensor dimensions: (?, 16, 16, 16)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
            # output tensor dimensions: (?, 32, 16, 16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output channels = 32
            # output tensor dimensions: (?, 32, 8, 8)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer
            self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return (SimpleCNN,)


@app.cell
def _(F, nn):
    class EnhancedCNN(nn.Module):
        def __init__(self):
            super(EnhancedCNN, self).__init__()
            # Convolutional Layer 1 with BatchNorm
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization after Conv1
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Convolutional Layer 2 with BatchNorm
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization after Conv2

            # Third convolutional layer
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output channels = 64
            self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization after Conv3

            # Fourth convolutional layer
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output channels = 128
            self.bn4 = nn.BatchNorm2d(128)  # Batch Normalization after Conv4

            # Fully connected layers with Dropout
            self.fc1 = nn.Linear(128 * 2 * 2, 128)
            self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            # First convolutional layer
            x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BatchNorm -> ReLU -> Pool

            # Second convolutional layer
            x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv -> BatchNorm -> ReLU -> Pool

            # Third convolutional layer
            x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv -> BatchNorm -> ReLU -> Pool

            # Fourth convolutional layer
            x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv -> BatchNorm -> ReLU -> Pool


            # Flatten the feature map
            x = x.view(-1, 128 * 2 * 2)

            # Fully connected layer 1 with Dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)

            # Fully connected layer 2 (output)
            x = self.fc2(x)
            return x
    return (EnhancedCNN,)


@app.cell
def _(EnhancedCNN):
    model = EnhancedCNN()
    print(model)
    return (model,)


@app.cell
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
        for i, data in enumerate(train_loader, 0):
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
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] accuracy: {correct/total:.3f}, loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")
    return (
        correct,
        criterion,
        data,
        epoch,
        i,
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
        The model has an **{accuracy:.2f}%** on the test set, much better than the accuracy we got from a fully connected network and due to parameter sharing of convolutional layers the CNN we constructed has less parameters!
        """
    )
    return


@app.cell
def _():
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

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for _i, idx in enumerate(indices):
        img = _images[idx].numpy().transpose((1, 2, 0))
        ax = fig.add_subplot(1, n_to_show, _i + 1)
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
    return (
        actual_single,
        ax,
        fig,
        idx,
        img,
        indices,
        n_to_show,
        preds,
        preds_single,
    )


if __name__ == "__main__":
    app.run()
