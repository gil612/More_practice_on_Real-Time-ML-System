run-dev:
#   copy dev.env .env
	poetry run python src/main.py

build:
	docker build -t trade_producer .

run: build
	docker run \
		--network=redpanda-network \
		--env KAFKA_BROKER_ADDRESS=redpanda:9092 \
		trade_producer
