FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb libgl1-mesa-dri libglib2.0-0 libegl1 fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /wizard
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3", "train_meta.py", "--scenarios", "all"]
