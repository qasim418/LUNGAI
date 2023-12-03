# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9.13

WORKDIR /lunngtestai
ADD . /lunngtestai

RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python-headless==4.5.3.56


RUN pip install -r requirements.txt
#RUN pip uninstall opencv-python



CMD [ "python" , "main.py" ]
RUN chown -R 42420:42420 /lunngtestai
ENV HOME=/lunngtestai