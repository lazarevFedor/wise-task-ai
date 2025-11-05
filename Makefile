genM: 
	protoc -I proto proto/llm-service/llm-service.proto \
	-I ./third_party/googleapis \
	--go_out=./llm-service/pkg/api --go_opt=paths=source_relative \
	--go-grpc_out=./llm-service/pkg/api --go-grpc_opt=paths=source_relative

genS:
	protoc -I proto proto/core-service/core-service.proto \
	--go_out=./server/pkg/api --go_opt=paths=source_relative \
	--go-grpc_out=./server/pkg/api --go-grpc_opt=paths=source_relative

