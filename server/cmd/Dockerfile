FROM golang:1.24

WORKDIR /app

COPY ../server .

RUN go build -o core ./cmd/main.go

ENTRYPOINT ["./core"]