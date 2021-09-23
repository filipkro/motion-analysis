FROM python:3.7.11-slim-bullseye

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.7 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"
RUN /home/myuser/venv/bin/python3.7 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app
