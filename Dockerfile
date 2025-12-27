FROM gsam2-base:latest

USER root

RUN useradd -m -s /bin/bash root && \
    mkdir -p /home/root/.cache /home/root/.config && \
    chown -R root:root /home/root

WORKDIR /home/root/workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=root:root . .
