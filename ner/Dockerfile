FROM python:3.8

RUN mkdir ner
WORKDIR ner

COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

RUN python3 -m deeppavlov install ner_ontonotes_bert_torch

EXPOSE 2801

CMD ["python", "app.py"]
