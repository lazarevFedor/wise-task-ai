package main

import (
	"context"
	"fmt"

	"github.com/lazarevFedor/wise-task-ai/server/internal/server"
	api "github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"
)


func main() {
	ctx := context.Background()
	log, err := logger.NewLogger(ctx)
	if err != nil {
		log.Error(ctx, "Failed to make new logger", zap.Error(err))
		return
	}

	container, port := "localhost", "0000"
	clientConn, err := grpc.NewClient(fmt.Sprintf("%s:%s", container, port), 
										grpc.WithTransportCredentials(insecure.NewCredentials()))

	if err != nil{
		log.Error(ctx, "failed to connect to LLM Service", zap.Error(err))
		return
	}

	client := api.NewCoreServiceClient(clientConn)

	server2LLM := server.NewServer(client)

	server := grpc.NewServer()

	api.RegisterCoreServiceServer(server, server2LLM)
	reflection.Register(server)

	// DB
}
