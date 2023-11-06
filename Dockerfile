FROM huggingface/transformers-pytorch-gpu:4.29.2
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install git -y

RUN mkdir -p /workspace 
WORKDIR /workspace 

COPY requirements.txt .
RUN pip install -r requirements.txt
