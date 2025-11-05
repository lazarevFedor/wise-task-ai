package server

import(
	"github.com/lazarevFedor/wise-task-ai/server/pkg/api"
	"context"
	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
)

type Server struct{
	api.UnimplementedAIServiceServer
		
}

func (s *Server) Prompt(ctx context.Context, req *api.PromptRequest) (*api.PromptResponse, error){
	
}
