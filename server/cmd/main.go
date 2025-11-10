package main

import (
	"context"
	"fmt"

	"github.com/lazarevFedor/wise-task-ai/server/internal/config"
	"github.com/lazarevFedor/wise-task-ai/server/internal/coreserver"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/llm-service"
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

	cfg, err := config.NewCoreServerConfig()

	container, port := "localhost", "0000"
	clientConn, err := grpc.NewClient(fmt.Sprintf("%s:%s", container, port),
		grpc.WithTransportCredentials(insecure.NewCredentials()))

	if err != nil {
		log.Error(ctx, "failed to connect to LLM Service", zap.Error(err))
		return
	}

	client := llm.NewLlmServiceClient(clientConn)

	server2LLM, err := coreserver.NewServer(ctx, client, cfg)

	//TODO: add arguments to grpcc.NewServer func
	server := grpc.NewServer()

	core.RegisterCoreServiceServer(server, server2LLM)
	reflection.Register(server)

	// DB
}
