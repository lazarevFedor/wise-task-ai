// Package graceful contains core server's graceful shutdown
package graceful

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)



type ShutDownFunc func(ctx context.Context) error

func Wait(ctx context.Context, grpcServer *grpc.Server, extra ...ShutDownFunc){

	log := logger.GetLoggerFromCtx(ctx)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	sig := <- sigChan
	log.Info(ctx, "Recieved shutdown signal", zap.String("signal", sig.String()))

	active := grpcServer.GetServiceInfo()
	log.Info(ctx, "Active services", zap.Any("services", active))

	shutdownCtx, cancel := context.WithTimeout(ctx, 5 * time.Second)
	defer cancel()

	done := make(chan struct{})

	go func(){
		grpcServer.GracefulStop()
		close(done)
	}()

	select{
	case <-done:
		log.Info(ctx, "gRPC server stoped gracefully")
	case <-shutdownCtx.Done():
		log.Warn(ctx, "Graceful stop timeout - forcing Stop()")
		grpcServer.Stop()
	}

	for _, fn := range extra{
		if err := fn(shutdownCtx); err != nil{
			log.Warn(ctx, "Shutdown step failed", zap.Error(err))
		}
	}
	log.Info(ctx, "gRPC server shutdown complete")
}