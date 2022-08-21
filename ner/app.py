from deeppavlov import configs, build_model
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import config
import json

app = FastAPI()

class Item(BaseModel):
	data: str

ner_model = build_model(configs.ner.ner_ontonotes_bert_mult_torch, download=True)

@app.post("/extract-entities")
def extract_entities(item: Item):
	
	ret_data = ner_model([item.data])
	
	result = []
	for itm in range(len(ret_data[0][0])):
		if ret_data[1][0][itm] != 'O':
			result.append({ret_data[1][0][itm]: ret_data[0][0][itm]})
			
	return {"response_code": "ok", "result": result}
	
if __name__ == '__main__':
	uvicorn.run(app, host=config.HOST, port=config.PORT)
	