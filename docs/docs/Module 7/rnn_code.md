# RNN Class Activity

As part of this activity we will update the text generation API developed in Module 3 to use an RNN. You will encorporate appropriate code from Module6-RNN notebook into your FastAPI docker implementation. 

---

## 1. Code Updates

Most of the activity code from Module 3 can be reused, but you need to modify the following lines:


```python
bigram_model = BigramModel(corpus)
```

Instead of the BigramModel you should use one of the autoregressive models (RNN, LSTM, GRU).

Similarly, replace the generate_text function/endpoint from the BigramModel with the analogous generate_with_rnn function/endpoint using the neural net.

```python
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = # TODO
    return {"generated_text": generated_text}
```

## 2. Test the Text Generation functionality

Rebuild the docker file from Module 3 and test to make sure that the /generate_with_rnn api endpoint works correctly.