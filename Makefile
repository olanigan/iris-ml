install:
	pip install -r requirements.txt

feature:
	python3 src/features.py

train:
	python3 src/training.py

all: install feature train