package main

import (
	"context"
	"fmt"
	"github.com/lazarevFedor/wise-task-ai/pkg/logger"
	pb "github.com/lazarevFedor/wise-task-ai/pkg/server_api"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

func main() {
	ctx := context.Background()
	log, err := logger.NewLogger(ctx)
	if err != nil {

	}
	container, Port := "name", "0000"
	conn, err := grpc.Dial(fmt.Sprintf("%s:%s", container, Port), grpc.Withinsecure())
	if err != nil {
		log.Error(ctx, "failed to connect to llm service", zap.Error(err))
		return
	}
	defer conn.Close()

}
