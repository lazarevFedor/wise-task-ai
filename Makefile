genGo:
	protoc -I proto proto/**/*.proto \
	-I third_party/googleapis \
	--go_out=./server/pkg/api --go_opt=paths=source_relative \
	--go-grpc_out=./server/pkg/api --go-grpc_opt=paths=source_relative

qdrant-build:
	docker compose -f ./docker/docker-compose.yml \
	run --rm qdrant_indexer python main.py --recreate --batch-size 64
	docker compose -f ./docker/docker-compose.yml up -d --build qdrant_db qdrant_ingest

build:
	docker compose -f ./docker/docker-compose.yml up -d --build postgresql_feedbacks
	docker compose -f ./docker/docker-compose.yml up -d --build migrator
	docker compose -f ./docker/docker-compose.yml --env-file .env up -d --build core_server

down:
	docker compose -f ./docker/docker-compose.yml down -v
