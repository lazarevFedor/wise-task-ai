// Package config contains service configuration
package config

import (
	"fmt"

	"github.com/ilyakaznacheev/cleanenv"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/db"
)

type CoreServerConfig struct {
	Qdrant   db.QdrantConfig   `env-prefix:"CORE_QDRANT_"`
	Postgres db.PostgresConfig `env-prefix:"CORE_POSTGRES_"`
	Host     string            `env:"CORE_SERVER_HOST"`
	IntPort  string            `env:"CORE_SERVER_INT_PORT"`
	RestPort string            `env:"CORE_SERVER_REST_PORT"`
}

func NewCoreServerConfig() (*CoreServerConfig, error) {
	var cfg CoreServerConfig
	if err := cleanenv.ReadEnv(&cfg); err != nil {
		return nil, fmt.Errorf("NewCoreServerConfig: failed to read env: %w", err)
	}
	return &cfg, nil
}
