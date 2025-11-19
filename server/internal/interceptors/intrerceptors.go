// Package interceptors cointains gRPC interceptors for coreService
package interceptors

import (
    "context"

    "github.com/google/uuid"
    "github.com/lazarevFedor/wise-task-ai/server/pkg/logger"
    "go.uber.org/zap"
    "google.golang.org/grpc"
    "google.golang.org/grpc/status"
)


func UnaryServerInterceptor(rootCtx context.Context) grpc.UnaryServerInterceptor {
    rootLogger := logger.GetLoggerFromCtx(rootCtx)

    return func(
        ctx context.Context,
        req interface{},
        info *grpc.UnaryServerInfo,
        handler grpc.UnaryHandler,
    ) (resp interface{}, err error) {

        reqID := uuid.NewString()
        ctx = logger.WithRequestID(ctx, reqID)

        ctx = logger.NewContextWithLogger(ctx, rootLogger)
		childLogger := logger.GetLoggerFromCtx(ctx)
        childLogger.Info(ctx, "Incoming gRPC request",
            zap.String("method", info.FullMethod),
        )

        resp, err = handler(ctx, req)

        if err != nil {
            st, _ := status.FromError(err)
            childLogger.Error(ctx, "gRPC request failed",
                zap.String("method", info.FullMethod),
                zap.String("error", st.Message()),
                zap.Any("code", st.Code()),
            )
        } else {
            childLogger.Info(ctx, "gRPC request completed",
                zap.String("method", info.FullMethod),
                zap.Any("response", resp),
            )
        }

        return resp, err
    }
}