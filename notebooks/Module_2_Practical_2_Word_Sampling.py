import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Module 2: Practice 2 - Word Sampling

    [![Open in molab](https://marimo.io/molab-shield.png)](https://molab.marimo.io/notebooks/nb_rDo2KD72hZNSLfQNybp5qw)
    """
    )
    
    return


if __name__ == "__main__":
    app.run()
