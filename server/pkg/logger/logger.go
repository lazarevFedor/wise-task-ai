package logger

import(
	"context"
	"go.uber.org/zap"
)

const(
	RequestID = "request_id",
	Key = "Logger"
)

var logger *zap.Logger

type Logger struct{
	l *zap.Logger
}

func NewLogger(ctx context.Context) (*Logger, err){
	logger, err := zap.NewDevelopment()
	if err != nil{
		return nil, err
	}
	logger = &Logger{logger}
	ctx = context.WithValue(ctx, Key, logger)
	return logger, nil
}

func GetLoggerFromCtx(ctx context.Context) *Logger{
	return ctx.Value(Key).(*Logger)
}

func (l *Logger) Info(ctx contetx.Context, msg string, fields ...zap.Field){
	if ctx.Value(RequestID) != nil{
		fields = append(fields, zap.String(RequestID, ctx.Value(RequestID)))
	}
	l.l.Info(msg, fields...)
}

func (l *Logger) Error(ctx context.Context, msg string, fields ...zap.Field){
	if ctx.Value(RequestID) != nil{
		fields = append(fields, zap.String(RequsetID, ctx.Value(RequestID)))
	}
	l.l.Error(msg, fileds...)
}


