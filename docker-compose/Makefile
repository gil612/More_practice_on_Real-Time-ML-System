start-redpanda:
	docker compose -f redpanda.yml up -d

stop-redpanda:
	docker compose -f redpanda.yml down

start-feature-pipeline:
	docker compose -f feature-pipeline.yml up --build -d


stop-feature-pipeline:
	docker compose -f feature-pipeline.yml down

start-backfill-pipeline:
	docker compose -f backfill-pipeline.yml up --build -d

stop-backfill-pipeline:
	docker compose -f backfill-pipeline.yml down

build-topic-to-feature:
	docker compose -f feature-pipeline.yml build --no-cache topic_to_feature_store
	docker compose -f feature-pipeline.yml up -d topic_to_feature_store





