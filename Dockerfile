FROM tensorflow/tensorflow:2.3.3 
WORKDIR /code
ENV FLASK_APP=hw
ENV FLASK_RUN_HOST=0.0.0.0
COPY . . 
RUN pip install virtualenv
RUN virtualenv infere
RUN . infere/bin/activate
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
