all: create-venv install-simple-deps

create-venv:
	python3 -m venv venv
	@echo "Virtual environment created!"

jupyter:
	@venv/bin/jupyter lab

install-deps:
	venv/bin/pip install -r requirements.txt

install-simple-deps:
	venv/bin/pip install -r requirements_simple.txt

describe:
	@venv/bin/python3 src/describe_generator.py

train:
	@venv/bin/python3 src/train_model/train.py

visualize:
	@venv/bin/python3 src/data_plotting/visualize.py

evaluate:
	@venv/bin/python3 src/check_precision/evaluate.py