package server

import(
	"github.com/lazarevFedor/wise-task-ai/pkg/api"
)

type Server struct{
	api.UnimplementedAIServiceServer
	
}
