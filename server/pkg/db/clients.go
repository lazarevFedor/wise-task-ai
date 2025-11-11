package db

import (
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/qdrant/go-client/qdrant"
)

type Clients struct {
	Postgres *pgxpool.Pool
	Qdrant   *qdrant.Client
}
