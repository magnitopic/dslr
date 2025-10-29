all: create-venv install-deps

create-venv:
	python3 -m venv venv
	@echo "Virtual environment created!"

jupyter:
	@venv/bin/jupyter lab

install-deps:
	venv/bin/pip install -r requirements.txt

describe:
	@venv/bin/python3 src/V1.Data_Analysis/describe.py

train:
	@venv/bin/python3 src/V3.Logistic_Regression/logreg_train.py data/dataset_train.csv

accuracy:
	@venv/bin/python3 src/Bonus/accuracy.py data/houses.csv data/dataset_train.csv

