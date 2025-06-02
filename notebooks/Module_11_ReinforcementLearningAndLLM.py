import marimo

__generated_with = "0.11.22"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Module 11 - Applying Reinforcement Learning to Language

        The goal of this activity is to build intuition on how Reinforcement Learning can modify the behavior of a language model.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's first construct a simplified language model. Our model will have a set vocabulary size **vocab_size** and for simplicity we'll use a subset of **vocab_size** words from the vocabulary of english words in the **nltk** library .""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    import nltk
    import random

    # Download word list (only needs to be done once)
    nltk.download('words')

    from nltk.corpus import words

    def build_a_vocab(vocab_size=200, min_len=3, max_len=8, seed=42):
        all_words = words.words()
        filtered = [
            w.lower() for w in all_words
            if w.isalpha() and min_len <= len(w) <= max_len
        ]
        unique = sorted(set(filtered))
        random.seed(seed)
        selected = random.sample(unique, vocab_size)
        return selected

    vocab = build_a_vocab(vocab_size=1000)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    print("Sample vocab:", vocab[:10])
    print("Total words:", len(vocab))

    vocab_size = len(vocab)
    return (
        build_a_vocab,
        idx_to_word,
        nltk,
        nn,
        optim,
        random,
        torch,
        vocab,
        vocab_size,
        word_to_idx,
        words,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To simulate a language model generating text, we'll use a simple model that generates the next word starting with the last letter of the context. It will uniformly sample from all of the words in the vocabulary that satisfy this condition. We will also leave an epsilon probability that the model samples more generally from the whole vocabulary.""")
    return


@app.cell
def _(idx_to_word, random, torch, vocab, word_to_idx):
    def is_valid_transition(prev, curr):
        return curr[0] == prev[-1]
    
    def generate_text_from_base_model(context):
        admissible_idxs = [i for i, w in enumerate(vocab) if is_valid_transition(idx_to_word[context[-1]], w)]
        epsilon = 0.2
        if (random.random() < epsilon):
            action = word_to_idx[random.choice(vocab)]
        else:
            action = random.choice(admissible_idxs)
   
        # We also calculate the probabilities for later. This is slightly complicated, since an admissible word can be sampled both based on the uniform probability from the whole vocabulary (20% of the time (epsilon)) and from the admissible words 80% of the time.
        if (action in admissible_idxs):
            log_prob = torch.tensor(epsilon/len(vocab) + (1-epsilon)/len(admissible_idxs)).log()
        else:
            log_prob = torch.tensor(epsilon/len(vocab)).log()
        
        return action, log_prob
    return generate_text_from_base_model, is_valid_transition


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Feel free to expertiment with trying different context below:""")
    return


@app.cell
def _(generate_text_from_base_model, idx_to_word):
    context = [10, 23, 40]
    print("Context: ", [idx_to_word[ind] for ind in context])
    completion = []
    for i in range(5):
        next_word_index, log_prob = generate_text_from_base_model(context)
        context.append(next_word_index)
        completion.append(idx_to_word[next_word_index])
    print("Completion: ", ', '.join(completion))
    return completion, context, i, log_prob, next_word_index


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is the base model behavior. What happens if we want to modify it? Say we want to generate new text in such a way that the new words **always** follow the word chain rule and the last word in the chain ends in the letter **m**.

        We could train a new model or fine-tune the existing model, but that would require many examples of this behavior. Can we make our text generation behave in the way we want by simply having it experiment on its own? All we would need would be some rules to tell it if it is doing a good job. This is a perfect setup for using Reinforcement Learning.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's put this problem in an RL setting of states, actions, and rewards. The **state** will be the current context of length **context_size**.""")
    return


@app.cell
def _(torch, vocab_size):
    def encode_sequence_state(context, context_size=3):
        vec = torch.zeros(context_size * vocab_size, dtype=torch.float32)
        last_k = context[-context_size:] if len(context) >= context_size else context
        offset = context_size - len(last_k)

        for i, idx in enumerate(last_k):
            vec[(offset + i) * vocab_size + idx] = 1.0
        return vec

    return (encode_sequence_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        The **action** is simply choosing one of the words from the vocabulary.  

        The **reward** is where things get interesting. The idea is to shape it in a way to get the behavior we want.
        """
    )
    return


@app.cell
def _(is_valid_transition):
    def shaped_reward(sequence):
        reward = 0.0
        for i in range(1, len(sequence)):
            if is_valid_transition(sequence[i - 1], sequence[i]):
                reward += 1.0
            else:
                reward -= 3.0

        if sequence[-1][-1] == 'm':
            reward += 10
        return reward
    return (shaped_reward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How should the agent choose the next action (next word)? Here is an example of a policy that uses a simple neural network to output probabilities of choosing different words from the vocabulary.""")
    return


@app.cell
def _(nn):
    class PPOPolicy(nn.Module):
        def __init__(self, input_dim, hidden_dim, action_dim):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
            self.policy_head = nn.Linear(hidden_dim, action_dim)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, state):
            x = self.shared(state)
            return self.policy_head(x), self.value_head(x)

    return (PPOPolicy,)


@app.cell
def _(
    encode_sequence_state,
    generate_text_from_base_model,
    idx_to_word,
    is_valid_transition,
    optim,
    random,
    shaped_reward,
    torch,
    vocab,
    word_to_idx,
):
    import torch.nn.functional as F

    max_seq_len = 5

    def train_ppo(policy, end_word, episodes=3000):
        optimizer = optim.Adam(policy.parameters(), lr=0.005)
        epsilon = 0.2
        reward_history = []

        for ep in range(episodes):
            start_word = random.choice(vocab)
            start_idx = word_to_idx[start_word]
        
            context = [start_idx]
            log_probs = []
            values = []

            for t in range(max_seq_len - 1):
                state = encode_sequence_state(context).unsqueeze(0)
            
                logits, value = policy(state)
                logits = logits[0]

                prev_word = idx_to_word[context[-1]]
                admissible_idxs = [i for i, w in enumerate(vocab) if is_valid_transition(prev_word, w)]

                if admissible_idxs and random.random() > epsilon:
                    probs = F.softmax(logits[admissible_idxs], dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    sampled = dist.sample()
                    action = admissible_idxs[sampled.item()]
                    log_prob = dist.log_prob(sampled)
                else:
                    action, log_prob = generate_text_from_base_model(context)

                context.append(action)
                log_probs.append(log_prob)
                values.append(value.squeeze())

                if idx_to_word[action] == end_word:
                    break

            sequence = [idx_to_word[i] for i in context]
            reward = shaped_reward(sequence)
            reward_history.append(reward)
            rewards = [reward] * len(log_probs)

            values = torch.stack(values)
            returns = torch.tensor(rewards)
            advantages = returns - values.detach()
            new_log_probs = torch.stack(log_probs)
            policy_loss = -torch.mean(new_log_probs * advantages)
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ep % 100 == 0:
                # print("Start word: ", start_word)
                print(f"Ep {ep}: {' → '.join(sequence)} | Reward = {reward:.2f}")

        return reward_history
    return F, max_seq_len, train_ppo


@app.cell
def _(torch):
    import matplotlib.pyplot as plt

    def plot_rewards(rewards):
        plt.figure(figsize=(10, 4))
        plt.plot(rewards, alpha=0.4, label="Raw reward")
        if len(rewards) >= 100:
            smoothed = torch.tensor(rewards).unfold(0, 100, 1).mean(1)
            plt.plot(range(99, len(rewards)), smoothed, label="Smoothed (100 ep)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress with PPO")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return plot_rewards, plt


@app.cell
def _(PPOPolicy, plot_rewards, random, train_ppo, vocab, vocab_size):
    end_word = random.choice(vocab)
    input_dim = input_dim = 3 * vocab_size # max_seq_len * vocab_size
    policy = PPOPolicy(input_dim, hidden_dim=64, action_dim=vocab_size)

    rewards = train_ppo(policy, end_word, episodes=1500)
    plot_rewards(rewards)
    return end_word, input_dim, policy, rewards


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The model is now trained and we can generate new text! Note, how our reward shape affected the generation. Feel free to experiment with increasing vocabulary size, maximum sequence length and the reward shape (e.g. can we reward the model for making longer or shorter chains or having some other interesting structure.)""")
    return


@app.cell
def _(
    encode_sequence_state,
    idx_to_word,
    is_valid_transition,
    random,
    torch,
    vocab,
    word_to_idx,
):
    def generate_chain(policy, start_word, end_word, max_len=5, epsilon_bad=0.0):
        context = [word_to_idx[start_word]]

        for _ in range(max_len - 1):
            state = encode_sequence_state(context).unsqueeze(0)
        
            logits, _ = policy(state)
            logits = logits[0]

            prev_word = idx_to_word[context[-1]]
            admissible_idxs = [i for i, w in enumerate(vocab) if is_valid_transition(prev_word, w)]

            if admissible_idxs and random.random() > epsilon_bad:
                probs = torch.softmax(logits[admissible_idxs], dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = admissible_idxs[dist.sample().item()]
            else:
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            context.append(action)

            if idx_to_word[action] == end_word:
                break

        return [idx_to_word[i] for i in context]

    return (generate_chain,)


@app.cell
def _(end_word, generate_chain, max_seq_len, policy, random, vocab):
    start_word = random.choice(vocab)
    print(start_word, end_word)

    chain = generate_chain(policy, start_word, end_word, max_len=max_seq_len)
    print(" → ".join(chain))
    return chain, start_word


if __name__ == "__main__":
    app.run()
