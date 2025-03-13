install:
	pip install --upgrade pip&&\
	pip install -r requirements.txt

test:
	python -m pytest -vv tests/test00-run.py

example01: 
	python macros/example01-plot-photons.py

example02:
	python macros/example02-train-photons.py

all: install test