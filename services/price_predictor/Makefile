.PHONY: install run training

install:
	poetry install

training-dev:
	poetry run python src/training.py || exit 0

prediction-dev:
	poetry run python src/prediction.py || exit 0


build:
	docker build -t price_predictor_training .

run: build
	docker run --env-file .env price_predictor_training