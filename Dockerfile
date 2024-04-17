FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install OpenJDK (Java) alongside Python
RUN apt-get update && \
    apt-get install -y default-jdk

ENV JAVA_HOME /usr/lib/jvm/default-java

WORKDIR /app

COPY resources/pyterrier_jars/ ~/
COPY resources/pyterrier_jars/ /root/

COPY dashboard_llm/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -U langchain-community
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install -i https://pypi.org/simple/ bitsandbytes

COPY dashboard_llm/ ./

CMD python -u llm_api.py
