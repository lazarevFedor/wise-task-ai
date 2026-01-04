package main

import (
	"context"
	"fmt"
	"net"
	"net/http"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"github.com/lazarevFedor/wise-task-ai/server/internal/config"
	"github.com/lazarevFedor/wise-task-ai/server/internal/coreserver"
	"github.com/lazarevFedor/wise-task-ai/server/internal/graceful"
	"github.com/lazarevFedor/wise-task-ai/server/internal/interceptors"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/llm-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/db"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"
)

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
			w.Header().Set("Access-Control-Max-Age", "86400")
			w.Header().Set("Vary", "Origin")
		}

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func main() {

	// Logger
	rootCtx := context.Background()
	rootCtx, err := logger.NewLoggerContext(rootCtx, true)
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

	llmClient := llm.NewLlmServiceClient(conn)

	log.Info(rootCtx, "Server starting at:",
		zap.String("IntHost", cfg.Host),
		zap.String("IntPort", cfg.IntPort),
		zap.String("RestPort", cfg.RestPort))

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", cfg.IntPort))
	if err != nil {
		log.Error(rootCtx, "failed to start core-server listening", zap.Error(err))
		return
	}
	server := grpc.NewServer(
		grpc.UnaryInterceptor(interceptors.ContextInterceptor(rootCtx)),
	)

	coreServer := coreserver.NewServer(llmClient, *dbClients)

	core.RegisterCoreServiceServer(server, coreServer)

	reflection.Register(server)

	go func() {
		log.Info(rootCtx, "Server is listening")
		if err = server.Serve(lis); err != nil {
			log.Error(rootCtx, "Failed to launch server", zap.Error(err))
		}
	}()

	// REST Gateway

	mux := runtime.NewServeMux()
	muxWithCORS := corsMiddleware(mux)
	opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	err = core.RegisterCoreServiceHandlerFromEndpoint(rootCtx, mux, cfg.Host+":"+cfg.IntPort, opts)
	if err != nil {
		log.Error(rootCtx, "failed to register endpoint in REST gateway", zap.Error(err))
		return
	}

	go func() {
		log.Info(rootCtx, "REST Gateway starting at port:", zap.String("RestPort", cfg.RestPort))
		err = http.ListenAndServe(":"+cfg.RestPort, muxWithCORS)
		if err != nil {
			log.Error(rootCtx, "Server exited with error:", zap.Error(err))
		}
	}()

	// Graceful Shutdown

	var postgresDBCloser = func(ctx context.Context) error {
		log := logger.GetLoggerFromCtx(ctx)
		log.Info(ctx, "Postgres Client is closing")
		pgClient.Close()
		return nil
	}

	var llmConnCloser = func(ctx context.Context) error {
		log := logger.GetLoggerFromCtx(ctx)
		log.Info(ctx, "LLM Client connection is closing")
		if err := conn.Close(); err != nil {
			return fmt.Errorf("failed to close llm's connention: %w", err)
		}
		return nil
	}

	graceful.Wait(
		rootCtx,
		server,
		postgresDBCloser,
		llmConnCloser,
	)
}
