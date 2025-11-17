FROM golang:1.24 AS builder

WORKDIR /app

COPY ../server/go.mod ./ 
COPY ../server/go.sum ./
RUN go mod download

COPY ../server .

RUN go build -o core ./cmd/main.go


FROM debian:bookworm-slim
WORKDIR /app

COPY --from=builder /app/core ./
COPY ../db_data /app/db_data

ENTRYPOINT ["./core"]
