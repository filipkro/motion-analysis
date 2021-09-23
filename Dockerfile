FROM python:3.7.11-slim-bullseye
RUN apt-get update && apt-get install wget git gcc g++ libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.7 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"
RUN /home/myuser/venv/bin/python3.7 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir wheel
RUN pip3 install numpy==1.19.5
RUN echo lol
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install mmcv-full==1.1.5
RUN python -c "import mmcv; print(mmcv.__file__)"

ENV PYTHONUNBUFFERED 1

COPY . /app
