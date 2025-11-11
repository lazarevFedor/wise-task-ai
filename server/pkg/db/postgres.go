// Package db contains DB's connection
package db

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
)

type PostgresConfig struct {
	Host     string `env:"HOST"`
	Port     int    `env:"PORT"`
	DB       string `env:"DB"`
	Username string `env:"USERNAME"`
	Password string `env:"PASSWORD"`
	MaxConns int    `env:"MAXCONNS"`
	MinConns int    `env:"MINCONNS"`
}

func NewPostgres(ctx context.Context, cfg PostgresConfig) (*pgxpool.Pool, error) {
	// urlExample := "postgres://username:password@localhost:5432/database_name"
	connstring := fmt.Sprintf("postgres://%s:%s@%s:%d/%s",
		cfg.Username,
		cfg.Password,
		cfg.Host,
		cfg.Port,
		cfg.DB)

	pgPool, err := pgxpool.New(ctx, connstring)
	if err != nil {
		return nil, fmt.Errorf("NewPostgres: failed to create pool: %w", err)
	}

	log := logger.GetLoggerFromCtx(ctx)
	log.Info(ctx, "connected to postgres feedback db")

	return pgPool, nil
}
