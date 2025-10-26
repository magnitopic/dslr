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
	@venv/bin/python3 src/describe.py

train:
	@venv/bin/python3 src/logreg_train.py

