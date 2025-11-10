// Package qdrantrepository is repository layer for qdrant db
package qdrantrepository

import (
	"context"
	"fmt"

	"github.com/qdrant/go-client/qdrant"
)

const (
	collection = "latex_books"
)

type UnimplementedCoreRepository interface {
	Search(ctx context.Context, request string) ([]*qdrant.ScoredPoint, error)
}

type QdrantRepository struct {
	QdrantClient *qdrant.Client
}

func NewRepository(client *qdrant.Client) *QdrantRepository {
	return &QdrantRepository{client}
}

func (repo *QdrantRepository) Search(ctx context.Context, requestVector []float32) ([]*qdrant.ScoredPoint, error) {
	result, err := repo.QdrantClient.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuery(requestVector...),
	})
	if err != nil {
		return nil, fmt.Errorf("Search: failed to search in Qdrant: %w", err)
	}
	return result, nil
}
