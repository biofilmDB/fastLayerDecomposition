# Use an official Python runtime as a parent image
FROM python:3.7

# Install git and GLPK, which are outside of Python requirements.txt
RUN apt-get update && apt-get install -y libglpk-dev
# Also install a CPU OpenCL implementation.
# UPDATE: pocl-opencl-icd doesn't implement enough of OpenCL. I get the error:
##        Device side queue is unimplemented (clCreateCommandQueueWithProperties.c:93)
# RUN apt-get update && apt-get install -y libglpk-dev pocl-opencl-icd

# Set the working directory to /app
WORKDIR /app

# Copy the code
COPY . /app/fastLayerDecomposition/

WORKDIR /app/fastLayerDecomposition

# Install Python dependencies via requirements.txt
RUN CVXOPT_BUILD_GLPK=1 pip3 install -r requirements.txt

# Run the server
WORKDIR /app/fastLayerDecomposition/image-layer-updating-GUI/

# Expose port 9988 for the websocket
EXPOSE 9988
# Expose port 8000 for the GUI
EXPOSE 8000

# Tell PyOpenCL to use device 0 and not prompt.
# ENV PYOPENCL_CTX='0'

CMD sh runboth.sh

## Create fastlayer image by running the following next to the Dockerfile:
# docker build -t fastlayer https://github.com/CraGL/fastLayerDecomposition
## Run fastlayer with:
# docker run -p 8000:8000 -p 9988:9988 --rm -it fastlayer
## Open your web browser to: http://localhost:8000/
