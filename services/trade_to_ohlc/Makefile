run-live-dev:
	copy live.dev.env .env
	poetry run python src/main.py

.PHONY: run-historical-dev run-historical-prod run-historical run-live

run-historical-dev:
	@echo "Running in historical dev mode..."
	copy /Y historical.dev.env .env
	del /f /q state\* 2>nul || true
	@echo "Environment file contents:"
	type .env
	poetry run python src/main.py

run-historical-prod:
	@echo "Running in historical prod mode..."
	copy /Y historical.prod.env .env
	del /f /q state\* 2>nul || true
	@echo "Environment file contents:"
	type .env
	poetry run python src/main.py

build:
	docker build -t trade_to_ohlc .

# For running in Docker with redpanda network
run-live: build
	docker run \
		--network=redpanda_network \
		--env KAFKA_BROKER_ADDRESS=redpanda:9092 \
		--env-file live.prod.env \
		trade_to_ohlc

# For running in Docker with redpanda network
run-historical: build
	@echo "Running historical mode in Docker..."
	docker run \
		--network=redpanda_network \
		--env KAFKA_BROKER_ADDRESS=redpanda:9092 \
		--env-file historical.prod.env \
		trade_to_ohlc


