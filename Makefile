gen: 
	protoc -I proto proto/llm-service/llm-service.proto \
	-I ./third_party/googleapis \
	--go_out=./llm-service/pkg/api --go_opt=paths=source_relative \
	--go-grpc_out=./llm-service/pkg/api --go-grpc_opt=paths=source_relative
