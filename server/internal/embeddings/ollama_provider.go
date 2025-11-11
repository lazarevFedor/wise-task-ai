package embeddings

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

const (
	model = "nomic-embed-text"
)

var (
	ollamaURL = "http://ollama:11434/api/embeddings"
)

type ollamaRequest struct {
	EmbedModel string `json:"model"`
	Text       string `json:"prompt"`
}

type ollamaResponse struct {
	Embedding []float64 `json:"embedding"`
}

func Embed(requestText string) ([]float32, error) {
	reqBody, _ := json.Marshal(ollamaRequest{
		EmbedModel: model,
		Text:       requestText,
	})

	resp, err := http.Post(ollamaURL, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("Embed: failed to request ollama server: %w", err)
	}
	defer resp.Body.Close()

	var result ollamaResponse

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("Emded: failed to Decode ollama response: %w", err)
	}

	vec := make([]float32, len(result.Embedding))
	for i, v := range result.Embedding {
		vec[i] = float32(v)
	}

	return vec, nil
}
