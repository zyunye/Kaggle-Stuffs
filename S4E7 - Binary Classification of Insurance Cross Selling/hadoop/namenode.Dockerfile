FROM apache/hadoop:3 as base

# RUN sudo yum update -y
RUN sudo yum install -y python3

RUN pip3 install pyspark jupyterlab