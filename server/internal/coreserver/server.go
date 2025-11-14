// Package coreserver contains core server settings and functionality
package coreserver

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/lazarevFedor/wise-task-ai/server/internal/config"
	"github.com/lazarevFedor/wise-task-ai/server/internal/repository/postgresrepository"
	"github.com/lazarevFedor/wise-task-ai/server/internal/qdrantservice"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/llm-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/db"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
)

type Server struct {
	core.UnimplementedCoreServiceServer
	llmClient    llm.LlmServiceClient
	postgresRepo *postgresrepository.PostgresRepository
}

func NewServer(ctx context.Context, client llm.LlmServiceClient, cfg *config.CoreServerConfig) (*Server, error) {
	postgresClient, err := db.NewPostgres(ctx, cfg.Postgres)
	if err != nil {
		return nil, fmt.Errorf("NewServer: failed to create postgres client: %w", err)
	}
	postgresRepo := postgresrepository.New(postgresClient)

	return &Server{llmClient: client,
		postgresRepo: postgresRepo,
	}, nil
}

func (s *Server) Prompt(ctx context.Context, req *core.PromptRequest) (*core.PromptResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	log := logger.GetLoggerFromCtx(ctx)


	seacrhResult, err := qdrantrservice.Search(req.Text)
	if err != nil {
		return nil, fmt.Errorf("Prompt: failed to search in Qdrant: %w", err)
	}

	log.Info(ctx, "Sending Qdrant's response to LLM...:", zap.Strings("requests", seacrhResult))
	llmResp, err := s.llmClient.Generate(ctx, &llm.GenerateRequest{
		Question: req.Text,
		Contexts: seacrhResult,
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
