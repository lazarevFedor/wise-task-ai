// Package postgresrepository is repository layer for postgres
package postgresrepository

import (
	"context"
	_ "embed"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/lazarevFedor/wise-task-ai/server/internal/entities"
)

var (
	//go:embed sql/insert_rate.sql
	insertRateRequest string

	//go:embed sql/insert_rate.sql
	getRatesRequest string
)

type PostgresRepository struct {
	pg *pgxpool.Pool
}

func NewRepository(client *pgxpool.Pool) *PostgresRepository {
	return &PostgresRepository{client}
}

func (repo *PostgresRepository) InsertRate(ctx context.Context, feedback *entities.Feedback) error {
	_, err := repo.pg.Exec(ctx, insertRateRequest,
		feedback.Request,
		feedback.Response,
		feedback.Mark)
	if err != nil {
		return fmt.Errorf("InsertRate: failed to insert rate to postgres: %w", err)
	}
	return nil
}

func (repo *PostgresRepository) GetRates(ctx context.Context) ([]*entities.Feedback, error) {
	rows, err := repo.pg.Query(ctx, getRatesRequest)
	if err != nil {
		return nil, fmt.Errorf("GetRates: failed to get rates from postgres: %w", err)
	}
	defer rows.Close()

	rates := make([]*entities.Feedback, 0)

	for rows.Next() {
		feedback := &entities.Feedback{}
		err = rows.Scan(&feedback.ID,
			&feedback.Request,
			&feedback.Response,
			&feedback.Mark,
			&feedback.CreatedAt)
		if err != nil {
			return nil, fmt.Errorf("GetRates: Scan error: %w", err)
		}
		rates = append(rates, feedback)
	}

	return rates, nil
}
