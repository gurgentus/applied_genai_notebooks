import marimo

__generated_with = "0.11.22"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Module 2: Activity 1 - Probability

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Treasure Hunt Game

        To get comfortable with handling probability expressions, we will play a treasure hunt game on a 5x5 grid.
        The rules are simple:

        - There is a treasure hidden somewhere on the grid (💎).
        - You have to guess where it is in a minimum number of tries.
        - Every time you guess I give you a distance clue: the manhattan distance to the treasure.
        - To make this a bit more challenging:
            - 80% of the time the clue will be correct.
            - 10% of the time it will be off by 1.
            - 10% of the time it will be off by -1.
        - Click on different cells below to start the game!
        - Click on 'Simulate 1 Turn' to choose cell based on the highest probability and update the probability table based on the latest distance clue.
        """
    )
    return


@app.cell(hide_code=True)
def _(buttons, field, game, mo, prob):
    mo.md(fr'''

    {mo.md('YOU WIN' if game.treasure_location == game.current_guess else f"Distance clue:  {game.distance_clue}").callout(kind="success")}

    {field}   

    {buttons}

    Probability Table (only updated when Simulating Turns):

    {prob} 


    Guesses: {game.seeker_guesses}
    ''')
    # Treasure Location: {game.treasure_location}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Explanation

        How does the 'Simulate 1 Turn' work to find the location in only a few tries?

        Can we use probability theory to make this efficient?

        Note that I gave you the probabilities of what the distance clue should be, given the treasure location. We want to *reverse* that probability and obtain the probability of treasure location given the distance clue. This is a perfect application of Bayes Theorem!

        Notation:

        Let's use $T_{ij}$ as the random variable that describes the existence of treasure in cell $(i,j)$.

        We will say that $T_{ij} = 🌱$ if there is no treasure in that cell and $T_{ij} = 💎$ if there is.

        After each turn we get a distance clue. Let $d_t$ represent the distance clue we get at turn $t$. 

        So $P(T_{ij} = 💎 | d_1)$ represents our belief after 1 turn that $T_{ij}$ contains treasure.

        $P(T_{ij} = 💎  | d_t, ..., d_2, d_1)$ represents our belief after $t$ turns ($t$ clues) that the cell $(i,j)$ contains treasure.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    number_array = mo.ui.array([mo.ui.number(value = None) for i in range(5)])
    mo.md(
        f"""
        ---
        ## Derivation

        Now let's add the information we have

        Let's try to calculate the different probabilities involved. 

        Say the treasure is in location $(1,2)$ and we guessed $(1,4)$ on our first turn. 
        What is the probability that the distance clue will be given as 0 (check the rules of the game in the beginning)?

        $P(d_1 = 0 | T_{{ij}} = 💎) =$ {number_array[0]}
        """
    )
    return (number_array,)


@app.cell(hide_code=True)
def _(mo, number_array):
    mo.md(
        f'''
        $P(d_1 = 0 | T_{{ij}} = 💎) = {number_array[0].value}$
        {'✅' if number_array[0].value == 0 
        else '❌'}

        {'Correct! The probability that the distance if off by 2 units is actually 0.' if number_array[0].value == 0 else 'Note that the real distance between those locations is 2, so what is the probability that the clue is off by 2 units?'} 
        '''
    )
    return


@app.cell(hide_code=True)
def _(mo, number_array):
    mo.md(
        f'''
        What about the probability that the distance clue will be given as 1?

        $P(d_1 = 1 | T_{{ij}} = 💎) =$ {number_array[1]} &nbsp;&nbsp; $P(d_1 = -1 | T_{{ij}} = 💎) =$ {number_array[2]}

        $P(d_1 = 2 | T_{{ij}} = 💎) =$ {number_array[3]} &nbsp;&nbsp; $P(d_1 = -2 | T_{{ij}} = 💎) =$ {number_array[4]}
        ''')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can now write a more general formula of the probability of the distance clue being given as $d$ if  the treasure is in cell (i,j) and our guess was cell (k,l):

        $$
        P(d_t = d | T_{ij} = 💎) = 
        \begin{cases}
        0.8 & \text{if distance((k,l), (i,j)) = d} \\
        0.2 & \text{if distance((k,l), (i,j)) = d $\pm 1$} \\
        0 & \text{otherwise}
        \end{cases}
        $$

        And using Bayes Formula:

            $$P(T_{ij} = 💎 | d_t = d, d_{t-1} ..., d_1) = \frac{\overbrace{P(d_t = d| T_{ij} = 💎, d_{t-1}, ..., d_1)}^{likelyhood} \overbrace{P(T_{ij} = 💎, d_{t-1}, ..., d_1)}^{prior}}{P(d_1, d_2, ..., d_{t})}$$

        Note how the prior gets updated after each new clue by being multiplied by the likelyhood. We don't have to worry about the denominator since it is the same for all of the cells, so as long as we normalize the probabilities at the end we are good to go.

        ----
        ## Implementation

        The code for the game is below. Most of the code deals with the setup of the game. The only part that we care about is the logic of how the algorithm decides where to search next based on the probability table update that uses the previous guess and the distance clue received after the previous guess. For each cell it calculates the distance to the  guess location (i.e. the correct distance if that cell had the treasure) and compares it with the distance clue received. This is the 'update_probabilities' function. It is just the implementation of the above formula.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ```python
        def update_probabilities(self, guess, distance_clue):
            new_grid = np.zeros_like(self.probability_grid)
            total_prob = 0

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    estimated_distance = abs(i - guess[0]) 
                        + abs(j - guess[1])
                    if estimated_distance == distance_clue: 
                        # correct clue
                        likelihood = 0.6
                    elif ((estimated_distance - distance_clue) == 1 
                        or (estimated_distance - distance_clue) == -1): 
                        # off by +1 or -1
                        likelihood = 0.2
                    else:
                        likelihood = 0
                    new_grid[i, j] = self.probability_grid[i, j] * likelihood
                    total_prob += new_grid[i, j]

            if total_prob > 0:
                new_grid /= total_prob
            self.probability_grid = new_grid
        ```
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    class BayesianTreasureHunt:
        def __init__(self, grid_size=5):
            self.grid_size = grid_size
            self.reset_game()

        def reset_game(self):
            self.treasure_location = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.probability_grid = np.full((self.grid_size, self.grid_size), 1 / (self.grid_size**2))
            self.seeker_guesses = []
            self.current_guess = None
            self.distance_clue = None
            self.iterations = 0

        def provide_clue(self, guess):
            """ Provides a noisy clue based on a distribution around the true distance with 20% incorrect information """
            """ Provides a noisy clue based on a distribution around the true distance """
            true_distance = abs(guess[0] - self.treasure_location[0]) + abs(guess[1] - self.treasure_location[1])
            noise = np.random.choice([0, -1, 1], p=[0.6, 0.2, 0.2]) 
            return max(0, true_distance + noise)

        def update_probabilities(self, guess, distance_clue):
            """ Update the probability grid using Bayesian inference """
            new_grid = np.zeros_like(self.probability_grid)
            total_prob = 0

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    estimated_distance = abs(i - guess[0]) + abs(j - guess[1])
                    if estimated_distance == distance_clue: # correct clue
                        likelihood = 0.6
                    elif ((estimated_distance - distance_clue) == 1 or 
                         (estimated_distance - distance_clue) == -1): # off by +1 or -1
                        likelihood = 0.2
                    else:
                        likelihood = 0
                    new_grid[i, j] = self.probability_grid[i, j] * likelihood
                    total_prob += new_grid[i, j]

            if total_prob > 0:
                new_grid /= total_prob
            self.probability_grid = new_grid

        def play_round_manual(self, guess):
            self.current_guess = guess
            self.seeker_guesses.append(self.current_guess)
            distance = self.provide_clue(self.current_guess)
            self.iterations += 1
            self.distance_clue = distance
            return distance

        def play_round(self):
            """ Plays one round of the game """
            self.current_guess = tuple(np.unravel_index(np.argmax(self.probability_grid, axis=None), self.probability_grid.shape))
            self.seeker_guesses.append(self.current_guess)
            if (self.current_guess == self.treasure_location):
                return 0
            else:
                self.probability_grid[self.current_guess] = 0
            distance_clue = self.provide_clue(self.current_guess)
            self.update_probabilities(self.current_guess, distance_clue)
            self.iterations += 1
            self.distance_clue = distance_clue
            return distance_clue

        def get_game_state(self):
            return {
                "Treasure Location": self.treasure_location[::-1],
                "Seeker Guesses": np.flip(self.seeker_guesses, axis=1),
                "Iterations": self.iterations,
                "Current Guess": self.current_guess[::-1]
            }

    # Run an interactive session
    game = BayesianTreasureHunt(grid_size=5)
    return BayesianTreasureHunt, game, np, plt


@app.cell(hide_code=True)
def _(game, mo):
    # Grid Button Code

    ui_grid = mo.ui.dictionary({
        f"{i,j}": mo.ui.button(label=f"{i,j}", value=(i,j),
            on_click= lambda value : game.play_round_manual(value)) 
                for i in range(game.grid_size) for j in range(game.grid_size)
    })
    button = mo.ui.button(label="Simulate 1 Turn", value="", kind="success", on_click= lambda value : game.play_round())
    reset_button = mo.ui.button(value=None, label="Reset Game", on_click= lambda value : game.reset_game())
    return button, reset_button, ui_grid


@app.cell(hide_code=True)
def _(button, game, reset_button, ui_grid):
    # Grid Graphics Code

    field = "|"
    for j in range(game.grid_size):
        field = field + "         |"
    field = field + "\n|"
    for j in range(game.grid_size):
        field = field + "-------- |"
    field = field + "\n"
    for i in range(game.grid_size):
        field = field + "| "
        for j in range(game.grid_size):
            field = field + ("💎" if ((i,j) in game.seeker_guesses) and ((i,j) == game.treasure_location)
                             else "🌱" if (i,j) in game.seeker_guesses else f"{ui_grid[f'({i}, {j})']}")
            field = field + " |"
        field = field + "\n"

    prob = "|"
    for j in range(game.grid_size):
        prob = prob + "         | "
    prob = prob + "\n| "
    for j in range(game.grid_size):
        prob = prob + "-------- | "
    prob = prob + "\n| "
    for i in range(game.grid_size):
        for j in range(game.grid_size):
            prob = prob + str(round(game.probability_grid[i,j], 2)) + " |"
        prob = prob + "\n|"

    buttons = f"{reset_button}" + " " + (f"{button}" if game.treasure_location != game.current_guess else "")
    return buttons, field, i, j, prob


if __name__ == "__main__":
    app.run()
