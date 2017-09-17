import json
import sys
def parseArguments():
	with open('src/hyperparams.json') as json_file:
		hp = json.load(json_file)
	with open('src/environment.json') as json_file:
		env = json.load(json_file)
	with open('src/train_params.json') as json_file:
		tp = json.load(json_file)
	#with open('src/test_params.json') as json_file:
	#	pp = json.load(json_file)
	with open('src/util_params.json') as json_file:
		up = json.load(json_file)
	return hp, tp, up, env
