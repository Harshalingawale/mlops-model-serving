.PHONY: install train serve test docker
install:
	pip install -r requirements.txt
train:
	python -m src.train
serve:
	uvicorn src.serve:app --reload
test:
	pytest -q
docker:
	docker build -t mlops-serving . && docker run -p 8000:8000 mlops-serving
