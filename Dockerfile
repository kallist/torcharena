FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TORCHARENA_HOME=/workspace/.torcharena
WORKDIR /app
COPY pyproject.toml README.md ./
COPY torcharena ./torcharena
COPY examples ./examples
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && useradd --create-home arena && mkdir /workspace && chown arena:arena /workspace
USER arena
WORKDIR /workspace
CMD ["torcharena", "doctor"]
