FROM python:3.11-slim

WORKDIR /app

# Install uv (pinned version for dependency-groups support)
RUN pip install --no-cache-dir uv>=0.5.0

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install only runtime dependencies (exclude dev dependencies like jupyter, mkdocs, etc.)
RUN uv sync --frozen --no-group dev && \
    uv run python -m spacy download en_core_web_lg    

# Copy all practical notebooks and make them read-only
COPY notebooks/Module_1_Practical_1_Probability.py ./
COPY notebooks/Module_1_Practical_2_Word_Sampling.py ./
COPY notebooks/Module_1_Practical_3_Word_Embeddings.py ./
COPY notebooks/Module_3_Practical_1_FCNN.py ./
COPY notebooks/Module_3_Practical_2_Convolutions.py ./
COPY notebooks/Module_3_Practical_3_CNN.py ./
COPY notebooks/Module_4_Practical_VAE.py ./
COPY notebooks/Module_5_Practical_1_GAN.py ./
COPY notebooks/Module_5_Practical_2_RNN.py ./
COPY notebooks/Module_7_Practical_1_EnergyBasedMethods.py ./
COPY notebooks/Module_7_Practical_2_DiffusionMethods.py ./
COPY notebooks/Module_8_Practical_GPT.py ./
COPY notebooks/Module_9_Practical_ReinforcementLearning.py ./
COPY notebooks/Module_Module_12_Practical_Music_Transformer.py ./

# Create non-root user and set ownership
RUN useradd -m -u 1000 marimo && \
    chown -R marimo:marimo /app && \
    chmod 444 Module_*_Practical*.py

# Switch to non-root user
USER marimo

# Expose port for marimo
EXPOSE 8080

# Run marimo server with shell to expand PORT env var (edit mode but files are read-only)
# Only load notebooks on demand by using file server mode
CMD ["sh", "-c", "uv run marimo edit --host 0.0.0.0 --port ${PORT:-8080} --no-token --headless"]
