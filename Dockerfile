# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
# Set the working directory in the container
WORKDIR /workspace
# Copy the current requirements file to the container
COPY requirements.txt .
# Ensure /opt/conda install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Clean any unneeded files like cache
RUN rm -rf /root/.cache/pip
