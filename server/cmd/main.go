package main

import (
	"context"
	"fmt"
	"net"

	
	"github.com/lazarevFedor/wise-task-ai/server/internal/config"
	"github.com/lazarevFedor/wise-task-ai/server/internal/interceptors"
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

func main() {

	// Logger
	rootCtx := context.Background()
	rootCtx, err := logger.NewLoggerContext(rootCtx)
	log := logger.GetLoggerFromCtx(rootCtx)
	if err != nil {
		log.Error(rootCtx, "Failed to make new logger", zap.Error(err))
		return
	}

	// Config
	cfg, err := config.NewCoreServerConfig()
	if err != nil {
		log.Error(rootCtx, "failed to load core configuration", zap.Error(err))
		return
	}

	// DB Connections
	var dbClients *db.Clients
	pgClient, err := db.NewPostgres(rootCtx, cfg.Postgres)
	if err != nil {
		log.Error(rootCtx, "failed to connect to Postgres", zap.Error(err))
		return
	}
	defer pgClient.Close()

	dbClients = &db.Clients{
		Postgres: pgClient,
	}

	// gRPC
	llmConnURL := fmt.Sprintf("%s:%s", cfg.LLMServer.Host, cfg.LLMServer.Port)
	conn, err := grpc.NewClient(llmConnURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Error(rootCtx, "failed to connect to LLM Service", zap.Error(err))
		return
	}
	defer conn.Close()

	llmClient := llm.NewLlmServiceClient(conn)

	log.Info(rootCtx, "Server starting at: ",
		zap.String("IntHost", cfg.Host),
		zap.String("IntPort", cfg.IntPort),
		zap.String("RestPort", cfg.RestPort))

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", cfg.RestPort))
	if err != nil {
		log.Error(rootCtx, "failed to start core-server listening", zap.Error(err))
		return
	}
	server := grpc.NewServer(
		grpc.UnaryInterceptor(interceptors.UnaryServerInterceptor(rootCtx)),
	)

	coreServer, err := coreserver.NewServer(llmClient, *dbClients)
	if err != nil {
		log.Error(rootCtx, "failed to create coreServer", zap.Error(err))
	}

	core.RegisterCoreServiceServer(server, coreServer)

	reflection.Register(server)

	log.Info(rootCtx, "Server is listening")
	if err = server.Serve(lis); err != nil {
		log.Error(rootCtx, "Failed to launch server", zap.Error(err))
	}
}
