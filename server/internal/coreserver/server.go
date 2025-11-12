// Package coreserver contains core server settings and functionality
package coreserver

import (
	"context"
	"fmt"
	"time"

	"github.com/lazarevFedor/wise-task-ai/server/internal/embeddings"
	"github.com/lazarevFedor/wise-task-ai/server/internal/entities"
	"github.com/lazarevFedor/wise-task-ai/server/internal/repository/postgresrepository"
	"github.com/lazarevFedor/wise-task-ai/server/internal/repository/qdrantrepository"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/llm-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/db"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	"go.uber.org/zap"
)

type Server struct {
	core.UnimplementedCoreServiceServer
	llmClient    llm.LlmServiceClient
	qdrantRepo   *qdrantrepository.QdrantRepository
	postgresRepo *postgresrepository.PostgresRepository
}

func NewServer(client llm.LlmServiceClient, dbCLients db.Clients) (*Server, error) {
	qdrantRepo := qdrantrepository.NewRepository(dbCLients.Qdrant)

	postgresRepo := postgresrepository.New(dbCLients.Postgres)

	return &Server{llmClient: client,
		qdrantRepo:   qdrantRepo,
		postgresRepo: postgresRepo}, nil
}

func (s *Server) Prompt(ctx context.Context, req *core.PromptRequest) (*core.PromptResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	log := logger.GetLoggerFromCtx(ctx)

	requestVector, err := embeddings.Embed(req.Text)
	if err != nil {
		return nil, fmt.Errorf("Prompt: failed to vectorize request: %w", err)
	}

	seacrhResult, err := s.qdrantRepo.Search(ctx, requestVector)
	if err != nil {
		return nil, fmt.Errorf("Prompt: failed to search in Qdrant: %w", err)
	}

	log.Info(ctx, "Sending Prompt to LLM...:", zap.Strings("requests", seacrhResult))
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
	feedback := &entities.Feedback{
		Request:  req.Prompt,
		Response: req.Response,
		Mark:     req.Mark,
	}
	resp := &core.FeedbackResponse{}
	if err := s.postgresRepo.InsertRate(ctx, feedback); err != nil {
		resp.Error = fmt.Sprintf("failed to insert rate to postgres db: %w", err)
		return resp, fmt.Errorf("failed to insert rate to postgres db: %w", err)
	}
	resp = &core.FeedbackResponse{Error: "OK"}
	return resp, nil
}
