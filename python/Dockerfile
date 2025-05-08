FROM python:3.13.3-bullseye


#RUN pip install https://github.com/intersystems-community/intersystems-irispython/releases/download/3.9.2/intersystems_iris-3.9.2-py3-none-any.whl
RUN pip install pandas sqlalchemy-iris==0.10.5 sentence_transformers
RUN pip install intersystems-irispython

WORKDIR /home/python/work

COPY ./work/*.py .

RUN python -c "import os; print(os.getcwd()); print(os.listdir())"
#RUN python create_desc_vectors.py

CMD ["/bin/bash", "-c", "python create_desc_vectors.py && sleep infinity"]
