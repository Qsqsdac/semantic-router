// Package lineartoken provides a compact binary token classifier for CPU gates.
package lineartoken

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"
)

// Model is the portable format exported from a binary CountVectorizer model.
type Model struct {
	ModelType     string    `json:"model_type"`
	Tokens        []string  `json:"tokens"`
	Weights       []float64 `json:"weights"`
	Intercept     float64   `json:"intercept"`
	NegativeLabel string    `json:"negative_label"`
	PositiveLabel string    `json:"positive_label"`
}

// Classifier evaluates one binary presence feature for each configured token.
type Classifier struct {
	model     Model
	automaton *automaton
}

// New validates model assets and prepares a matcher for request-time inference.
func New(model Model) (*Classifier, error) {
	if len(model.Tokens) == 0 {
		return nil, fmt.Errorf("linear token model has no tokens")
	}
	if len(model.Tokens) != len(model.Weights) {
		return nil, fmt.Errorf("linear token model has %d tokens but %d weights", len(model.Tokens), len(model.Weights))
	}
	for _, token := range model.Tokens {
		if strings.TrimSpace(token) == "" {
			return nil, fmt.Errorf("linear token model contains an empty token")
		}
	}
	return &Classifier{model: model, automaton: newAutomaton(boundaryPatterns(model.Tokens))}, nil
}

// Load reads an exported model from JSON.
func Load(path string) (*Classifier, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read linear token model: %w", err)
	}
	var model Model
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("parse linear token model: %w", err)
	}
	return New(model)
}

// Score returns the signed linear decision value.
func (c *Classifier) Score(text string) float64 {
	tokens := tokenizeLowerWords(text)
	if len(tokens) == 0 {
		return c.model.Intercept
	}
	seen := make([]bool, len(c.model.Tokens))
	c.automaton.findMatches(" "+strings.Join(tokens, " ")+" ", seen)
	score := c.model.Intercept
	for index, matched := range seen {
		if matched {
			score += c.model.Weights[index]
		}
	}
	return score
}

// Predict returns the label, whether the positive class matched, and its probability.
func (c *Classifier) Predict(text string) (string, bool, float64) {
	score := c.Score(text)
	positive := score > 0
	label := c.model.NegativeLabel
	if positive {
		label = c.model.PositiveLabel
	}
	return label, positive, 1 / (1 + math.Exp(-score))
}

func tokenizeLowerWords(input string) []string {
	runes := []rune(strings.ToLower(input))
	tokens := make([]string, 0, len(runes)/4)
	start := -1
	flush := func(end int) {
		if start >= 0 && end-start >= 2 {
			tokens = append(tokens, string(runes[start:end]))
		}
		start = -1
	}
	for index, char := range runes {
		if char == '_' || unicode.IsLetter(char) || unicode.IsDigit(char) {
			if start < 0 {
				start = index
			}
			continue
		}
		flush(index)
	}
	flush(len(runes))
	return tokens
}

func boundaryPatterns(tokens []string) []string {
	patterns := make([]string, len(tokens))
	for index, token := range tokens {
		patterns[index] = " " + strings.ToLower(token) + " "
	}
	return patterns
}

type node struct {
	next map[rune]int
	fail int
	out  []int
}

type automaton struct{ nodes []node }

func newAutomaton(patterns []string) *automaton {
	automaton := &automaton{nodes: []node{{next: map[rune]int{}}}}
	for index, pattern := range patterns {
		state := 0
		for _, char := range pattern {
			next, exists := automaton.nodes[state].next[char]
			if !exists {
				next = len(automaton.nodes)
				automaton.nodes[state].next[char] = next
				automaton.nodes = append(automaton.nodes, node{next: map[rune]int{}})
			}
			state = next
		}
		automaton.nodes[state].out = append(automaton.nodes[state].out, index)
	}
	queue := make([]int, 0)
	for _, state := range automaton.nodes[0].next {
		queue = append(queue, state)
	}
	for len(queue) > 0 {
		state := queue[0]
		queue = queue[1:]
		for char, next := range automaton.nodes[state].next {
			failure := automaton.nodes[state].fail
			for failure != 0 {
				if _, exists := automaton.nodes[failure].next[char]; exists {
					break
				}
				failure = automaton.nodes[failure].fail
			}
			if target, exists := automaton.nodes[failure].next[char]; exists {
				automaton.nodes[next].fail = target
			}
			fallback := automaton.nodes[next].fail
			automaton.nodes[next].out = append(automaton.nodes[next].out, automaton.nodes[fallback].out...)
			queue = append(queue, next)
		}
	}
	return automaton
}

func (a *automaton) findMatches(text string, seen []bool) {
	state := 0
	for _, char := range text {
		for state != 0 {
			if _, exists := a.nodes[state].next[char]; exists {
				break
			}
			state = a.nodes[state].fail
		}
		if next, exists := a.nodes[state].next[char]; exists {
			state = next
		} else {
			state = 0
		}
		for _, index := range a.nodes[state].out {
			seen[index] = true
		}
	}
}
