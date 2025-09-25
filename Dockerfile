FROM tensorflow/tensorflow:latest-gpu

# prevent python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps if needed (e.g. for numpy, scipy, umap, bokeh)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mount your code later, so only keep requirements in the image
