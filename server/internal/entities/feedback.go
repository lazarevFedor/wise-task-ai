// Package entities contains ptoject entitites
package entities

import (
	"time"
)

type Feedback struct {
	ID        int
	Request   string
	Response  string
	Mark      bool
	CreatedAt time.Time
}
