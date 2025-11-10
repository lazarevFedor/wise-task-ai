// Package db contains DB's connection
package db

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
)

type PostgresConfig struct {
	host     string `env:"CORE_POSTGRES_HOST"`
	port     int    `env:"CORE_POSTGRES_PORT"`
	db       string `env:"CORE_POSTGRES_DB"`
	username string `env:"CORE_POSTGRES_USERNAME"`
	password string `env:"CORE_POSTGRES_PASSWORD"`
	maxConns int    `env:"CORE_POSTGRES_MAXCONNS"`
	minConns int    `env:"CORE_POSTGRES_MINCONNS"`
}

func NewPostgres(ctx context.Context, cfg PostgresConfig) (*pgxpool.Pool, error) {
	// urlExample := "postgres://username:password@localhost:5432/database_name"
	connstring := fmt.Sprintf("postgres://%s:%s@%s:%d/%s",
		cfg.username,
		cfg.password,
		cfg.host,
		cfg.port,
		cfg.db)

	pgPool, err := pgxpool.New(ctx, connstring)
	if err != nil {
		return nil, fmt.Errorf("NewPostgres: failed to create pool: %w", err)
	}

	log := logger.GetLoggerFromCtx(ctx)
	log.Info(ctx, "connected to postgres feedback db")

	return pgPool, nil
}
