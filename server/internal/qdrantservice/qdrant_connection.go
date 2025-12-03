// Package qdrantservice is db and core interection
package qdrantservice

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

type qdrantRequest struct {
	Query string `json:"query"`
	Limit string `json:"limit"`
}

type qdrantResponse struct {
	Query      string         `json:"query"`
	Collection string         `json:"collection"`
	Results    []qdrantResult `json:"results"`
	Count      int            `json:"count"`
}

type qdrantResult struct {
	ID      int           `json:"id"`
	Score   float64       `json:"score"`
	Payload qdrantPayload `json:"payload"`
}

type qdrantPayload struct {
	Title      string `json:"title"`
	Source     string `json:"source"`
	ChunkIndex int    `json:"chunk_index"`
	Text       string `json:"text"`
}

type QdrantHealthCheckResponse struct {
	Status string `json:"status"`
}

func Search(prompt, limit string) ([]string, error) {

	if limit == "" {
		limit = os.Getenv("SEARCH_DEFAULT_LIMIT")
		if limit == "" {
			return nil, fmt.Errorf("failed to get limit var from env")
		}
	}

	QdrantURL := os.Getenv("QDRANT_INGEST_URL")
	if QdrantURL == "" {
		return nil, fmt.Errorf("failed to get QdrantURL var from env")
	}

	if err := CheckHealth(QdrantURL); err != nil {
		return nil, fmt.Errorf("seacrh: Qdrant is unhealth: %w", err)
	}

	reqBody, _ := json.Marshal(qdrantRequest{
		Query: prompt,
		Limit: limit,
	})
	req, _ := http.NewRequest("POST", QdrantURL+"/v1/search", bytes.NewBuffer(reqBody))
	req.Header.Set("Content-Type", "application/json")
	if key := os.Getenv("API_KEY"); key != "" {
		req.Header.Set("X-API-Key", key)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search: failed to request qdrant db: %w", err)
	}

	var response qdrantResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("search: failed to decode qdrant response: %w", err)
	}

	result := make([]string, 0)
	for _, res := range response.Results {
		result = append(result, res.Payload.Text)
	}

	return result, nil
}

func CheckHealth(QdrantURL string) error {
	req, _ := http.NewRequest("GET", QdrantURL+"/health", nil)
	if key := os.Getenv("API_KEY"); key != "" {
		req.Header.Set("X-API-Key", key)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("CheckHealth: failed to request qdrant db: %w", err)
	}

	var response QdrantHealthCheckResponse
	if err = json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return fmt.Errorf("ChecktHealth: failed to decode response: %w", err)
	}

	// FIXME: healthcheck endpoint returns "error" if error and "status" if alls ok, idk how to diff those situations
	if response.Status != "ok" {
		return fmt.Errorf("CheckHealth: Qdarant is unhealth, Status: %s", response.Status)
	}
	return nil
}
