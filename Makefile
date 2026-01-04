genGo:
	protoc -I proto proto/**/*.proto \
	-I third_party/googleapis \
	-I third_party/grpc-gateway \
	--openapiv2_out ./swagger \
	--grpc-gateway_out=./server/pkg/api --grpc-gateway_opt=paths=source_relative \
	--go_out=./server/pkg/api --go_opt=paths=source_relative \
	--go-grpc_out=./server/pkg/api --go-grpc_opt=paths=source_relative

genPy:
	python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./llm-service/app \
    --grpc_python_out=./llm-service/app \
    --pyi_out=./llm-service/app \
    ./proto/llm-service/llm-service.proto

down:
	docker-compose -f ./docker/docker-compose.yml down -v

build:
	docker-compose -f docker/docker-compose.yml --env-file .env up -d --build llama_cpp
	docker-compose -f docker/docker-compose.yml --env-file .env up -d --build llm_server

	docker-compose -f docker/docker-compose.yml run --rm qdrant_indexer python indexer.py --data-dir /data/latex_books --recreate --bm25-index /app/data/bm25_index.pkl
	docker-compose -f docker/docker-compose.yml up -d --build qdrant_ingest

	docker-compose -f docker/docker-compose.yml up -d --build postgresql_feedbacks
	docker-compose -f docker/docker-compose.yml up -d --build migrator
	docker-compose -f docker/docker-compose.yml --env-file .env up -d --build core_server

llm:
	docker-compose -f docker/docker-compose.yml --env-file .env up -d --build llama_cpp
	docker-compose -f docker/docker-compose.yml --env-file .env up -d --build llm_server

qdrant:
	docker-compose -f docker/docker-compose.yml run --rm qdrant_indexer python indexer.py --data-dir /data/latex_books --recreate --bm25-index /app/data/bm25_index.pkl
	docker-compose -f docker/docker-compose.yml up -d --build qdrant_indexer qdrant_ingest --no-cache

core:
	docker-compose -f docker/docker-compose.yml up -d --build postgresql_feedbacks
	docker-compose -f docker/docker-compose.yml up -d --build migrator
	docker-compose -f docker/docker-compose.yml --env-file .env up -d --build core_server

