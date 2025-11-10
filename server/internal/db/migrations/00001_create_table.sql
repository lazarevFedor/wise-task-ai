-- +goose Up
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS feedback(
    id int NOT NULL PRIMARY KEY,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    mark BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP TABLE IF EXISTS feedback;
-- +goose StatementEnd
