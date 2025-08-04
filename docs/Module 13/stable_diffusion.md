# Deploying Stable Diffusion Activity

As part of this activity you will deploy a Stable Diffusion model from HuggingFace. 

---

## 1. Choose a Hugging Face Model:

Use any variant of Stable Diffusion hosted on Hugging Face.

```python
# Load the model (use CPU for simplicity; change to 'cuda' for GPU)
model_id = # TODO
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # Change to "cuda" if using GPU
```

## 2. Update the API

Add a new function/endpoint called "/generate_with_stable_diffusion" to the existing endpoints that were defined in the previous activities.

```python
class ImageGenerationRequest(BaseModel):
    prompt: str

@app.post("/generate_with_stable_diffusion")
def generate_with_stable_diffusion(request: ImageGenerationRequest):
    image = pipe(data.prompt).images[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image_base64": img_str}
```

## 3. Test the image generation functionality

Rebuild the docker file and test to make sure that the /generate_with_stable_diffusion api endpoint works correctly.