FROM grounded_sam2:1.0

USER root

RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/.cache /home/appuser/.config && \
    chown -R appuser:appuser /home/appuser

WORKDIR /home/appuser/workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

USER appuser

COPY --chown=appuser:appuser . .
