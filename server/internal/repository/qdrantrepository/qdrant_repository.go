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

func (repo *QdrantRepository) Search(ctx context.Context, requestVector []float32) ([]string, error) {
	queryResponce, err := repo.QdrantClient.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuery(requestVector...),
	})
	if err != nil {
		return nil, fmt.Errorf("Search: failed to search in Qdrant: %w", err)
	}
	result := make([]string, 0)

	//TODO: result[0].Payload["Text"].GetStringValue()
	//TODO: result[0].String()
	for i, point := range queryResponce {
		result = append(result, point.String())
		if i == 3 {
			break
		}
	}
	return result, nil
}
