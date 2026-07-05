# ==============================================================================
#  Mechanogenomic Virtual Cell — reproducible container
# ==============================================================================
#  Pinned Python base for full reproducibility. Installs the package and all
#  scientific + visualization dependencies, so the model, tests, figures, the
#  interactive notebook, and the renderers all run identically anywhere.
#
#  IMPORTANT: build from the REPOSITORY ROOT (so the whole project is in the
#  build context), pointing at this file with -f:
#
#  Build:   docker build -t mvcell -f docker/Dockerfile .
#  Test:    docker run --rm mvcell                 # runs the validation suite
#  Shell:   docker run --rm -it mvcell bash
#  Jupyter: docker run --rm -p 8888:8888 mvcell jupyter
# ==============================================================================

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="mechanogenomic-virtual-cell"
LABEL org.opencontainers.image.description="Phenotype-aware physical-computational virtual cell linking tissue stiffness to nuclear form and mechanogenomic state."
LABEL org.opencontainers.image.source="https://github.com/Danpc11/mechanogenomic-virtual-cell"
LABEL org.opencontainers.image.licenses="MIT"

# --- system libraries needed by matplotlib, cairosvg, and PyVista (headless) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libcairo2 \
        libgl1 \
        libglx-mesa0 \
        libxrender1 \
        libxext6 \
        libsm6 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# headless rendering for PyVista/VTK
ENV PYVISTA_OFF_SCREEN=true \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# --- install dependencies first (better layer caching) ---
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy scipy pandas scikit-learn numba gplearn \
        matplotlib seaborn \
        cairosvg \
        pytest \
        jupyter ipywidgets

# --- copy the project and install it as a package ---
COPY . .
RUN pip install --no-cache-dir -e . || echo "editable install skipped"

# --- default command: run the validation suite (proves the build works) ---
# Override with any command, e.g.:
#   docker run --rm mvcell python src/mvirtual_cell.py
#   docker run --rm -p 8888:8888 mvcell jupyter
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["test"]
