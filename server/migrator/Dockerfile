FROM golang:1.23-alpine

RUN apk add --no-cache git

RUN go install github.com/pressly/goose/v3/cmd/goose@latest

WORKDIR /migrations

ENTRYPOINT [ "goose" ]