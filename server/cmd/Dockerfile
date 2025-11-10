FROM golang:1.24 AS builder

WORKDIR /app

COPY ../server .

RUN go build -o core ./cmd/main.go


FROM debian:bookworm-slim
WORKDIR /app

COPY --from=builder /app/core ./

# TODO: smth with db

RUN ls

ENTRYPOINT ["./core"]
