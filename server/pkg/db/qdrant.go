package db

import (
	"context"
	"fmt"

	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	"github.com/qdrant/go-client/qdrant"
)

type QdrantConfig struct {
	Host string `env:"CORE_QDRANT_HOST"`
	Port int    `env:"CORE_QDRANT_PORT"`
}

func NewQdrant(ctx context.Context, cfg *QdrantConfig) (*qdrant.Client, error) {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: cfg.Host,
		Port: cfg.Port,
	})
	if err != nil {
		return nil, fmt.Errorf("NewCoreReository: failed to connect to qdrant: %w", err)
	}

	log := logger.GetLoggerFromCtx(ctx)
	log.Info(ctx, "connected to qdrant db")

	return client, nil
}
