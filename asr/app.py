from time import sleep
import asr_handlers
import config
import json

import uvicorn
from fastapi import Request, Response, FastAPI

app = FastAPI()

@app.route('/asr_file_raw', methods=['POST'])
async def asr_file_raw(request: Request):
	raw_file = await request.body()
	file, length = asr_handlers.preproces_data(raw_file)
	
	cnt = 0
	num_worker = None
	while cnt < 600 / config.DELAY_WORKERS:
		num_worker = asr_handlers.manage_worker(None)
		if num_worker != None:
			break
			
		cnt = cnt + 1
		sleep(config.DELAY_WORKERS)
		
	if num_worker == None:
		return Response(content=json.dumps({"response_code": "error get asr worker"}), media_type="application/json")
	
	asr_handlers.WORKER_IN[num_worker].put(file)
	cnt = 0
	sum_frame = None
	sum_empty = True
	while cnt < 600 / config.DELAY_WORKERS:   
		if asr_handlers.WORKER_OUT[num_worker].empty() == False:
			sum_frame = asr_handlers.WORKER_OUT[num_worker].get()
			sum_empty = False
			break
		cnt = cnt + 1
		if cnt * config.DELAY_WORKERS / 1.0 > 0 and cnt * config.DELAY_WORKERS % 1.0 == 0:
			print("trying to get a worker again sec: " + str(cnt * config.DELAY_WORKERS / 1.0))
		sleep(config.DELAY_WORKERS)
		
	if asr_handlers.manage_worker(num_worker) != None:
		return Response(content=json.dumps({"response_code": "error release asr worker: " + str(num_worker)}), media_type="application/json")
		
	if sum_empty:
		return Response(content=json.dumps({"response_code": "error asr worker: " + str(num_worker) + " time out"}), media_type="application/json")
		
	if len(sum_frame) > 0:
		text, res, conf = asr_handlers.beam_decoder(sum_frame)
	else:
		text = ""
		res = []
		conf = 0
		
	num_norm = request.headers.get('num_norm')
	if num_norm != None and num_norm == "yes":
		text = asr_handlers.EXTRACTOR.replace(text, apply_regrouping=True)[0]
		
	punct = request.headers.get('punct')
	if punct != None and punct == "yes":
		text = asr_handlers.MODEL_PUNCT.enhance_text(text, lan='ru')
	
	return Response(content=json.dumps({"response_code": "ok", "result": [text, res, conf]}), media_type="application/json")
	
if __name__ == '__main__':
	uvicorn.run(app, host=config.HOST, port=config.PORT)
	