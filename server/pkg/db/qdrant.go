package db

import (
	"context"
	"fmt"

	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	"github.com/qdrant/go-client/qdrant"
)

type QdrantConfig struct {
	Host string `env:"HOST"`
	Port int    `env:"PORT"`
}

func NewQdrant(ctx context.Context, cfg QdrantConfig) (*qdrant.Client, error) {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: cfg.Host,
		Port: cfg.Port,
	})
	if err != nil {
		return nil, fmt.Errorf("NewCoreRepository: failed to connect to Qdrant: %w", err)
	}

	log := logger.GetLoggerFromCtx(ctx)
	log.Info(ctx, "connected to Qdrant db")
	return client, nil
}
