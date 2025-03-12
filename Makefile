install:
	pip install --upgrade pip&&\
	pip install -r requirements.txt

test:
	python -m pytest -vv tests/test00-run.py
	python -m pytest -vv tests/test01-plot-photons.py

lint:
	pylint --disable=R,C tests/*.py

all: install lint test