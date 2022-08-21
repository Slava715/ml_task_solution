Clone progect 
* git clone https://github.com/Slava715/ml_task_solution
  
From asr example api downald models on this link
* https://drive.google.com/file/d/1XFnxAnBJI2DJYwJYA7Ijba2kZLwa2jDy/view?usp=sharing
* Unpack models.tar.gz on ml_task_solution/asr folder
* tar -xvf models.tar.gz
  
	
Run asr example api 
* cd ml_task_solution/asr
* docker-compose build
* docker-compose up
  
	
Run ner example api
* cd ml_task_solution/ner
* docker-compose build
* docker-compose up
  
	
Run test scripts to see how it works
* pip3 install pydub
* pip3 install requests


	* python3 test_ner.py "Привет меня зовут Георгий. Мне наскучил ваш грустный мир. Поэтому сегодня вечером в 22 часа 32 минуты хочу улететь на Марс лежи на    диване. Ведь говорят, что на марсе классно."
  > * {'response_code': 'ok', 'result': [{'B-PERSON': 'Георгий'}, {'B-TIME': 'сегодня'}, {'B-TIME': 'вечером'}, {'B-TIME': '22'}, {'I-TIME': 'часа'}, {'I-TIME': '32'}, {'I-TIME': '   минуты'}, {'B-LOC': 'Марс'}]}
  
	
  * python3 test_asr.py test.wav wav
  > * {'response_code': 'ok', 'result': ['Привет меня зовут Георгий. Мне наскучил ваш грустный мир. Поэтому сегодня вечером в 22 часа 32 минуты хочу улететь на Марс лежи на диване.    Ведь говорят, что на марсе классно.', [{'start': 0.6, 'end': 1.24, 'word': 'привет'}, {'start': 1.4, 'end': 1.88, 'word': ' меня'}, {'start': 1.96, 'end': 2.6, 'word': ' зов   ут'}, {'start': 2.68, 'end': 3.68, 'word': ' георгий'}, {'start': 4.6, 'end': 5.08, 'word': ' мне'}, {'start': 5.08, 'end': 6.24, 'word': ' наскучил'}, {'start': 6.32, 'end':    6.72, 'word': ' ваш'}, {'start': 6.84, 'end': 7.76, 'word': ' грустный'}, {'start': 7.76, 'end': 8.2, 'word': ' мир'}, {'start': 9.36, 'end': 10.36, 'word': ' поэтому'}, {'s   tart': 10.44, 'end': 11.28, 'word': ' сегодня'}, {'start': 11.36, 'end': 12.24, 'word': ' вечером'}, {'start': 13.04, 'end': 13.52, 'word': ' в'}, {'start': 13.16, 'end': 14.   0, 'word': ' двадцать'}, {'start': 14.08, 'end': 14.44, 'word': ' два'}, {'start': 14.6, 'end': 15.16, 'word': ' часа'}, {'start': 15.92, 'end': 16.76, 'word': ' тридцать'},    {'start': 16.84, 'end': 17.2, 'word': ' две'}, {'start': 17.28, 'end': 18.04, 'word': ' минуты'}, {'start': 18.96, 'end': 19.6, 'word': ' хочу'}, {'start': 19.72, 'end': 20.6   , 'word': ' улететь'}, {'start': 20.64, 'end': 20.92, 'word': ' на'}, {'start': 20.96, 'end': 21.52, 'word': ' марс'}, {'start': 22.0, 'end': 22.76, 'word': ' лежи'}, {'start   ': 22.8, 'end': 23.04, 'word': ' на'}, {'start': 23.04, 'end': 23.8, 'word': ' диване'}, {'start': 24.68, 'end': 25.16, 'word': ' ведь'}, {'start': 25.2, 'end': 26.0, 'word':    ' говорят'}, {'start': 26.52, 'end': 26.92, 'word': ' что'}, {'start': 27.0, 'end': 27.28, 'word': ' на'}, {'start': 27.28, 'end': 27.96, 'word': ' марсе'}, {'start': 28.04,    'end': 28.84, 'word': ' классно'}], 0]}
  
	
  * python3 test_asr_ner.py test.wav wav
  > * {'response_code': 'ok', 'result': ['Привет меня зовут Георгий. Мне наскучил ваш грустный мир. Поэтому сегодня вечером в 22 часа 32 минуты хочу улететь на Марс лежи на диване.    Ведь говорят, что на марсе классно.', [{'start': 0.6, 'end': 1.24, 'word': 'привет'}, {'start': 1.4, 'end': 1.88, 'word': ' меня'}, {'start': 1.96, 'end': 2.6, 'word': ' зов   ут'}, {'start': 2.68, 'end': 3.68, 'word': ' георгий'}, {'start': 4.6, 'end': 5.08, 'word': ' мне'}, {'start': 5.08, 'end': 6.24, 'word': ' наскучил'}, {'start': 6.32, 'end':    6.72, 'word': ' ваш'}, {'start': 6.84, 'end': 7.76, 'word': ' грустный'}, {'start': 7.76, 'end': 8.2, 'word': ' мир'}, {'start': 9.36, 'end': 10.36, 'word': ' поэтому'}, {'s   tart': 10.44, 'end': 11.28, 'word': ' сегодня'}, {'start': 11.36, 'end': 12.24, 'word': ' вечером'}, {'start': 13.04, 'end': 13.52, 'word': ' в'}, {'start': 13.16, 'end': 14.   0, 'word': ' двадцать'}, {'start': 14.08, 'end': 14.44, 'word': ' два'}, {'start': 14.6, 'end': 15.16, 'word': ' часа'}, {'start': 15.92, 'end': 16.76, 'word': ' тридцать'},    {'start': 16.84, 'end': 17.2, 'word': ' две'}, {'start': 17.28, 'end': 18.04, 'word': ' минуты'}, {'start': 18.96, 'end': 19.6, 'word': ' хочу'}, {'start': 19.72, 'end': 20.6   , 'word': ' улететь'}, {'start': 20.64, 'end': 20.92, 'word': ' на'}, {'start': 20.96, 'end': 21.52, 'word': ' марс'}, {'start': 22.0, 'end': 22.76, 'word': ' лежи'}, {'start   ': 22.8, 'end': 23.04, 'word': ' на'}, {'start': 23.04, 'end': 23.8, 'word': ' диване'}, {'start': 24.68, 'end': 25.16, 'word': ' ведь'}, {'start': 25.2, 'end': 26.0, 'word':    ' говорят'}, {'start': 26.52, 'end': 26.92, 'word': ' что'}, {'start': 27.0, 'end': 27.28, 'word': ' на'}, {'start': 27.28, 'end': 27.96, 'word': ' марсе'}, {'start': 28.04,    'end': 28.84, 'word': ' классно'}], 0]}
  > * {'response_code': 'ok', 'result': [{'B-PERSON': 'Георгий'}, {'B-TIME': 'сегодня'}, {'B-TIME': 'вечером'}, {'B-TIME': '22'}, {'I-TIME': 'часа'}, {'I-TIME': '32'}, {'I-TIME': '   минуты'}, {'B-LOC': 'Марс'}]}
