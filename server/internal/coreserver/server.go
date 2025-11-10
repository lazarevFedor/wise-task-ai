// Package server contains core server settings and functionality
package coreserver

import (
	"context"
	"fmt"
	"go.uber.org/zap"
	"time"

	"github.com/lazarevFedor/wise-task-ai/server/internal/config"
	"github.com/lazarevFedor/wise-task-ai/server/internal/repository/postgresrepository"
	"github.com/lazarevFedor/wise-task-ai/server/internal/repository/qdrantrepository"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/llm-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/db"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
)

type Server struct {
	core.UnimplementedCoreServiceServer
	llmClient    llm.LlmServiceClient
	qdrantRepo   *qdrantrepository.QdrantRepository
	postgresRepo *postgresrepository.PostgresRepository
}

func NewServer(ctx context.Context, client llm.LlmServiceClient, cfg *config.CoreServerConfig) (*Server, error) {
	qdrantClient, err := db.NewQdrant(ctx, cfg.Qdrant)
	if err != nil {
		return nil, fmt.Errorf("NewServer: failed to create qdrant client: %w", err)
	}
	qdrantRepo := qdrantrepository.NewRepository(qdrantClient)

	postgresClient, err := db.NewPostgres(ctx, *cfg.Postgres)
	if err != nil {
		return nil, fmt.Errorf("NewServer: failed to create postgres client: %w", err)
	}
	postgresRepo := postgresrepository.NewRepository(postgresClient)

	return &Server{llmClient: client,
		qdrantRepo:   qdrantRepo,
		postgresRepo: postgresRepo,
	}, nil
}

func (s *Server) Prompt(ctx context.Context, req *core.PromptRequest) (*core.PromptResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	log := logger.GetLoggerFromCtx(ctx)

	//TODO: vectorize request text
	var requestVector []float32
	//FIXME: searchResult is not used, it waits for request vectorization
	seacrhResult, err := s.qdrantRepo.Search(ctx, requestVector)
	log.Error(ctx, "Prompt: FIXME: searchResult is not used", zap.Any("var", seacrhResult))
	if err != nil {
		return nil, fmt.Errorf("Prompt: failed to search in Qdrant: %w", err)
	}

	//TODO: unvectorize searchResult to []string
	var unvectorizedResult []string

	log.Info(ctx, "Sending Prompt to LLM...:")
	llmResp, err := s.llmClient.Generate(ctx, &llm.GenerateRequest{
		Question: req.Text,
		Contexts: unvectorizedResult,
	})
	if err != nil {
		return nil, fmt.Errorf("llmClient.Prompt: %w", err)
	}

	resp := &core.PromptResponse{Text: llmResp.Answer}
	return resp, nil
}

func (s *Server) Feedback(ctx context.Context, req *core.FeedbackRequest) (*core.FeedbackResponse, error) {
	log := logger.GetLoggerFromCtx(ctx)
	log.Info(ctx, "Sending Feedback to DB...:")
	// sending to db
	return nil, nil
}
