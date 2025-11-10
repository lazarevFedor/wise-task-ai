genGo:
	protoc -I proto proto/**/*.proto \
	-I third_party/googleapis \
	--go_out=./server/pkg/api --go_opt=paths=source_relative \
	--go-grpc_out=./server/pkg/api --go-grpc_opt=paths=source_relative
