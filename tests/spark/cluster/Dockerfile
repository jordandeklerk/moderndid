# Image of the standalone Spark cluster that compose.yaml runs. Since Spark Connect runs the client's Python functions
# on the workers, the image runs the client's Python minor version, 3.13. Because the official Spark images have
# Python 3.10 and no NumPy, this image installs its own Python environment on a Java 21 runtime.
FROM eclipse-temurin:21-jre-noble

COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    SPARK_HOME=/opt/venv/lib/python3.13/site-packages/pyspark \
    PYSPARK_PYTHON=/opt/venv/bin/python

RUN uv venv --python 3.13 /opt/venv && uv pip install "pyspark[connect]==4.2.0" "pandas<3"

# Since the dependencies change less often than the code, they get a layer of their own.
COPY pyproject.toml /src/
RUN uv pip install -r /src/pyproject.toml
COPY README.md LICENSE /src/
COPY moderndid /src/moderndid
RUN uv pip install --no-deps /src
