FROM nvcr.io/nvidia/nemo:22.05

RUN mkdir asr
WORKDIR asr

COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

RUN cd ctcdecode && pip install .

EXPOSE 2701

CMD python3 app.py
