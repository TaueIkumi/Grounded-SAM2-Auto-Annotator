FROM grounded_sam2:1.0

WORKDIR /home/appuser/workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

