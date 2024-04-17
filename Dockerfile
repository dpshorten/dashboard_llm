FROM nvcr.io/nvidia/pytorch:24.02-py3



ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install OpenJDK (Java) alongside Python
RUN apt-get update && \
    apt-get install -y default-jdk

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/default-java

# Copy the current directory contents into the container at /app
#COPY . /app
WORKDIR ~/
COPY llm_files/Llama-2-13b-chat-hf_27_3_24 Llama-2-13b-chat-hf_27_3_24

#RUN mkdir /root/.pyterrier
#COPY .pyterrier/ /root/.pyterrier/
#COPY Llama-2-13b-chat-hf /app/Llama-2-13b-chat-hf/
#COPY requirements.txt .
#COPY QAv4_api_v1.py .
#COPY Llama-2-13b-chat-hf .
#COPY CryoSat-2.txt .
#COPY Fengyun-2D.txt .
#COPY Fengyun-2E.txt .
#COPY Fengyun-2F.txt .
#COPY Fengyun-2H.txt .
#COPY Fengyun-4A.txt .
#COPY Haiyang-2A.txt .
#COPY Jason-1.txt .
#COPY Jason-2.txt .
#COPY Jason-3.txt .
#COPY SARAL.txt .
#COPY Sentinel-3A.txt .
#COPY Sentinel-3B.txt .
#COPY Sentinel-6A.txt .
#COPY Summary.txt .
#COPY TOPEX.txt .

COPY docker_llm/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -U langchain-community
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install -i https://pypi.org/simple/ bitsandbytes


COPY docker_llm/QAv4_api_v1.py .

COPY pyterrier_jars/ ~/
COPY pyterrier_jars/ /root/

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD python -u QAv4_api_v1.py
