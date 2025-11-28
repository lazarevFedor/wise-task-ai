// Package coreserver contains core server settings and functionality
package coreserver

import (
	"context"
	"fmt"
	"github.com/lazarevFedor/wise-task-ai/server/internal/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"time"

	"go.uber.org/zap"

	"github.com/lazarevFedor/wise-task-ai/server/internal/entities"
	"github.com/lazarevFedor/wise-task-ai/server/internal/qdrantservice"
	"github.com/lazarevFedor/wise-task-ai/server/internal/repository/postgresrepository"
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

func NewServer(client llm.LlmServiceClient, dbCLients db.Clients) *Server {
	postgresRepo := postgresrepository.New(dbCLients.Postgres)

	return &Server{llmClient: client,
		postgresRepo: postgresRepo,
	}
}

func (s *Server) Prompt(ctx context.Context, req *core.PromptRequest) (*core.PromptResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	log := logger.GetLoggerFromCtx(ctx)

	resp := &core.PromptResponse{}

	searchResult, err := qdrantservice.Search(req.Text)
	if err != nil {
		dualErr := errors.NewDualError(err, errors.SearchFailedErr)
		log.Error(ctx, "prompt: failed to search in Qdrant", zap.Error(dualErr.Internal()))
		return resp, status.Errorf(codes.Unavailable, "%s", dualErr.Public())
	}

	log.Debug(ctx, "Sending Qdrant's response to LLM...:", zap.Strings("requests", searchResult))
	llmResp, err := s.llmClient.Generate(ctx, &llm.GenerateRequest{
		Question:  req.Text,
		Contexts:  searchResult,
		RequestId: ctx.Value(logger.RequestID).(string),
	})
	if err != nil {
		dualErr := errors.NewDualError(err, errors.LLMUnavailableErr)
		log.Error(ctx, "prompt: failed to request llmClient.Generate", zap.Error(err))
		return resp, status.Errorf(codes.Unavailable, dualErr.Public())
	}

	resp = &core.PromptResponse{
		Text:           llmResp.Answer,
		ProcessingTime: llmResp.ProcessingTime,
	}
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
		dualErr := errors.NewDualError(err, errors.PSQLFailedErr)
		log.Error(ctx, "failed to insert rate to postgres db", zap.Error(err))
		return resp, status.Errorf(codes.Unavailable, "%s", dualErr.Public())
	}

	return resp, nil
}

func (s *Server) HealthCheck(ctx context.Context, req *core.HealthRequest) (*core.HealthResponse, error) {
	req = nil
	log := logger.GetLoggerFromCtx(ctx)

	resp := &core.HealthResponse{
		Healthy: false,
	}

	err := qdrantservice.CheckHealth()
	if err != nil {
		dualErr := errors.NewDualError(err, errors.CoreUnavailableErr)
		log.Error(ctx, "Core_HealthCheck: qdrant unhealth", zap.Error(dualErr.Internal()))
		return resp, status.Errorf(codes.Unavailable, "%s", dualErr.Public())
	}

	llmHealthResp, err := s.llmClient.HealthCheck(ctx, &llm.HealthRequest{})
	if err != nil {
		dualErr := errors.NewDualError(err, errors.LLMUnavailableErr)
		log.Error(ctx, "failed to get response from llm service", zap.Error(dualErr.Internal()))
		return resp, status.Errorf(codes.Unavailable, "%s", dualErr.Public())
	}

	if !llmHealthResp.Healthy {
		dualErr := errors.NewDualError(
			fmt.Errorf("HealthCheck: LLM service is unhealth: %w", err),
			errors.LLMUnhealthErr,
		)
		log.Error(ctx, "HealthCheck: LLM service is unhealth", zap.Error(dualErr.Internal()))
		return resp, status.Errorf(codes.Unavailable, "%s", dualErr.Public())
	}
	resp.Healthy = true
	return resp, nil
}
