package logger

import (
	"context"
	"fmt"

	"go.uber.org/zap"
)

const (
	RequestID = "request_id"
	Key       = "Logger"
)

type Logger struct {
	l *zap.Logger
}

func NewLogger(ctx context.Context) (*Logger, error) {
	logger, err := zap.NewDevelopment()
	if err != nil {
		return nil, fmt.Errorf("NewLogger: %w", err)
	}
	ctx = context.WithValue(ctx, Key, &Logger{logger})
	return ctx.Value(Key).(*Logger), nil
}

func GetLoggerFromCtx(ctx context.Context) *Logger {
	return ctx.Value(Key).(*Logger)
}

func (l *Logger) Info(ctx context.Context, msg string, fields ...zap.Field) {
	if ctx.Value(RequestID) != nil {
		fields = append(fields, zap.String(RequestID, ctx.Value(RequestID).(string)))
	}
	l.l.Info(msg, fields...)
}

func (l *Logger) Error(ctx context.Context, msg string, fields ...zap.Field) {
	if ctx.Value(RequestID) != nil {
		fields = append(fields, zap.String(RequestID, ctx.Value(RequestID).(string)))
	}
	l.l.Error(msg, fields...)
}
