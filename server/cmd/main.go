package main

import (
	"context"
	"fmt"
	"net"

	"github.com/lazarevFedor/wise-task-ai/server/internal/config"
	"github.com/lazarevFedor/wise-task-ai/server/internal/coreserver"

	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/llm-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/db"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"
)

const (
	llmHost = "llm-service"
	llmPort = "8081"
)

func main() {
	// Logger
	ctx := context.Background()
	ctx, err := logger.NewLoggerContext(ctx)
	log := logger.GetLoggerFromCtx(ctx)
	if err != nil {
		log.Error(ctx, "Failed to make new logger", zap.Error(err))
		return
	}

	// Config
	cfg, err := config.NewCoreServerConfig()
	if err != nil {
		log.Error(ctx, "failed to load core configuration", zap.Error(err))
		return
	}

	// DB Connections
	var dbClients *db.Clients
	pgClient, err := db.NewPostgres(ctx, cfg.Postgres)
	if err != nil {
		log.Error(ctx, "failed to connect to Postgres", zap.Error(err))
		return
	}
	defer pgClient.Close()

	dbClients = &db.Clients{
		Postgres: pgClient,
	}

	// gRPC
	llmConnURL := fmt.Sprintf("%s:%s", llmHost, llmPort)
	conn, err := grpc.NewClient(llmConnURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Error(ctx, "failed to connect to LLM Service", zap.Error(err))
		return
	}
	defer conn.Close()

	llmClient := llm.NewLlmServiceClient(conn)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", cfg.RestPort))
	if err != nil {
		log.Error(ctx, "failed to start core-server listening", zap.Error(err))
		return
	}

	//TODO: add arguments to grpc.NewServer func
	server := grpc.NewServer()
	coreServer, err := coreserver.NewServer(llmClient, *dbClients)
	if err != nil {
		log.Error(ctx, "failed to create coreServer", zap.Error(err))
	}

	core.RegisterCoreServiceServer(server, coreServer)

	if err = server.Serve(lis); err != nil {
		log.Error(ctx, "Failed to launch server", zap.Error(err))
	}

	reflection.Register(server)
}
