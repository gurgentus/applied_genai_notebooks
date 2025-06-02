import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Module 7: Practical - Normalizing Flows""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Normalizing flows solve the problem of evaluating the likelyhood function by modeling the mapping from the latent distribution using a form, whose likelihood and parameters are tractable to calculate.

        $z = f(x)$

        $p_{data}(x) = p_{latent}(z)|J(x)|$

        $Loss$ = 

        $-\mathbb{E}_{x \sim p_{data}} (\log(p_{data}(x))) =$

        $-\sum_{i=1}^N \log( p_{model}(x_i | w ) = -\sum_{i=1}^N \log\Big( p_{latent}(z_i | w ) |J(x_i)|\Big) =$

        $-\sum_{i=1}^N \Big( \log( p_{latent}(f(x_i | w )) + \log(|J(x_i)|) \Big)$
        """
    )
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import re

    # Load and preprocess Count of Monte Cristo
    url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"

    import requests
    text = requests.get(url).text

    # Keep only the main body (remove header/footer)
    start_idx = text.find("Chapter 1.")
    end_idx = text.rfind("Chapter 5.") # text.rfind("End of the Project Gutenberg")
    text = text[start_idx:end_idx]

    # Pre-processing
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()

    # Tokenization
    tokens = text.split()

    # Vocabulary construction
    from collections import Counter
    counter = Counter(tokens)

    # We'll assign indices 0 and 1 to special tokens "<PAD>" and "<UNK>", the rest of the indeces
    # are based on the frequency of the words.
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(9998))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    inv_vocab = {idx: word for word, idx in vocab.items()}

    # Encode tokens
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    return (
        Counter,
        DataLoader,
        Dataset,
        counter,
        encoded,
        end_idx,
        inv_vocab,
        nn,
        re,
        requests,
        start_idx,
        text,
        tokens,
        torch,
        url,
        vocab,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Since we are training the model to predict the next word in a sequence, we will construct our training set features based on 30 word sequences from the text. The corresponding labels are the sequences shifted by one word.""")
    return


@app.cell
def _(DataLoader, Dataset, encoded, torch):
    # Create sequences
    SEQ_LEN = 30
    class TextDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data) - SEQ_LEN

        def __getitem__(self, idx):
            return (torch.tensor(self.data[idx:idx+SEQ_LEN]),
                    torch.tensor(self.data[idx+1:idx+SEQ_LEN+1]))

    train_datasets = TextDataset(encoded)
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    return SEQ_LEN, TextDataset, train_datasets, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Let's see what the first pair of input/output sequences look like.""")
    return


@app.cell
def _(train_loader):
    next(iter(train_loader))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We now define the LSTM architecture. Since the LSTM layer is already implemented in PyTorch, we can use it directly. The LSTM layer takes the input sequence and returns the output sequence along with the hidden state. The output is then passed through a fully connected layer to get the final predictions.""")
    return


@app.cell
def _(nn, torch):
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, hidden=None):
            x = self.embedding(x)
            x, hidden = self.lstm(x, hidden)
            x = self.fc(x)
            return x, hidden

    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return LSTMModel, criterion, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, we train the model for 15 epochs. The loss should decrease over time, indicating that the model is learning to predict the next word in the sequence. After training, we can generate text by providing a seed phrase and letting the model predict the next words based on the learned patterns.""")
    return


@app.cell
def _(criterion, model, optimizer, train_loader):
    # Training loop
    for epoch in range(15):
        total_loss = 0
        for inputs, targets in train_loader:
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    return epoch, inputs, loss, outputs, targets, total_loss


@app.cell
def _(mo):
    mo.md(r"""Finally we generate new text by feeding a seed sentence to the model. We then""")
    return


@app.cell
def _(inv_vocab, torch, vocab):
    def generate_text(model, seed_text, length=50, temperature=1.0):
        model.eval()
        words = seed_text.lower().split()
        input_ids = [vocab.get(w, vocab["<UNK>"]) for w in words]
        input_tensor = torch.tensor(input_ids).unsqueeze(0)
        hidden = None

        with torch.no_grad():
            for _ in range(length):
                output, hidden = model(input_tensor, hidden)
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                words.append(inv_vocab.get(next_id, "<UNK>"))

                # Extend input sequence with new token
                input_ids.append(next_id)
                input_tensor = torch.tensor(input_ids).unsqueeze(0)

        return " ".join(words)
    return (generate_text,)


@app.cell
def _(generate_text, model):
    seed = "the count of monte cristo"
    print("\nGenerated Text:\n")
    print(generate_text(model, seed, length=50))
    return (seed,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
