package server

import (
	"context"
	"fmt"

	"github.com/lazarevFedor/wise-task-ai/server/pkg/api/core-service"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	)

type Server struct{	
	api.UnimplementedCoreServiceServer
	llmClient api.CoreServiceClient
}

func NewServer(client api.CoreServiceClient) (*Server){
	return &Server{ llmClient: client }
}

func (s *Server) Prompt(ctx context.Context, req *api.PromptRequest) (*api.PromptResponse, error){
	log := logger.GetLoggerFromCtx(ctx)
	log.Info(ctx, "Sending Prompt to LLM...:")
	resp, err := s.llmClient.Prompt(ctx, req)
	if err != nil{
		return nil, fmt.Errorf("llmClient.Prompt: %w", err)
	}
	return resp, nil
}



func (s *Server) Feedback(ctx context.Context, req *api.FeedbackRequest) (*api.FeedbackResponse, error){
	log := logger.GetLoggerFromCtx(ctx)
	log.Info(ctx, "Sending Feedback to DB...:")
	// sending to db
	return nil, nil
}


