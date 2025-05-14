ARG IMAGE=intersystems/iris-community:2025.1
FROM $IMAGE

WORKDIR /home/irisowner/dev

ARG TESTS=0
ARG MODULE="GenAI"
ARG NAMESPACE="USER"

## Embedded Python environment

# This is required for IPM
ENV IRISNAMESPACE="IRISAPP"

# This is not required. Why is here?
#ENV PYTHON_PATH=/usr/irissys/bin/
ENV PATH="/usr/irissys/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/irisowner/bin:/home/irisowner/.local/bin"


## Start IRIS
RUN --mount=type=bind,src=.,dst=. \
    # These are used in the creation of the image. requirements.txt is installed via IPPM
    pip3 install -r requirements-dev.txt --break-system-packages && \
    iris start IRIS && \
    iris merge IRIS merge.cpf && \
  # iris stop IRIS quietly restart && \
    irispython iris_script.py && \
    iris stop IRIS quietly
