package db

import (
	"github.com/jackc/pgx/v5/pgxpool"
)

type Clients struct {
	Postgres *pgxpool.Pool
}
