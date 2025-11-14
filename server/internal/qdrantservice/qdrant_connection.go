// Package qdrantservice is db and core interection
package qdrantrservice

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

const (
	QdrantURL  = "http://qdrant_ingest:8080"
)

type qdrantRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type qdrantResponse struct {
	Colection string          `json:"collection"`
	Query     string          `json:"query"`
	Results    []qdrantResults `json:"results"`
	Amount    int             `json:"count"`
}

type qdrantResults struct {
	Payload qdrantPayload `json:"payload"`
}

type qdrantPayload struct {
	Text string `json:"text"`
}

func Search(prompt string) ([]string, error) {
	if err := checkHealth(); err != nil {
		return nil, fmt.Errorf("Seacrh: Qdrant id unhealth: %w", err)
	}

	reqBody, _ := json.Marshal(qdrantRequest{
		Query: prompt,
		Limit: 5,
	})
	req, _ := http.NewRequest("POST", QdrantURL+"/v1/search", bytes.NewBuffer(reqBody))
	if key := os.Getenv("API-KEY"); key != "" {
		req.Header.Set("X-API-Key", key)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Search: failed to request qdrant db: %w", err)
	}

	var response qdrantResponse
	if err := json.NewDecoder(req.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("Search: failed to encode qdrant response: %w", err)
	}

	result := make([]string, 0)
	for _, res := range response.Results {
		result = append(result, res.Payload.Text)
	}

	return result, nil
}

type QdrantHealthCheckResponse struct{
	Status string `json:"status"`
}

func checkHealth() error {
	resp, err := http.Get(QdrantURL+"/health")
	if err != nil {
		return fmt.Errorf("CheckHealth: failed to request qdrant db: %w", err)
	}
	var response QdrantHealthCheckResponse
	if err = json.NewDecoder(resp.Body).Decode(&response); err != nil{
		return fmt.Errorf("ChecktHealth: failed to decode response: %w", err)
	}

	if response.Status != "ok"{
		return fmt.Errorf("CheckHealth: Qdarant is unhealth, Status: %s", response.Status)
	}
	return nil
}