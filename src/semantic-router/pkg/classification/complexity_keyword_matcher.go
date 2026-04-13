package classification

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type ComplexityKeywordMatcher interface {
	Classify(text string) (string, bool, error)
	Free()
}

type AhoComplexityKeywordMatcher struct {
	classifier *nlp_binding.AhoClassifier
}

type complexityKeywordMapping struct {
	Hard []string `json:"hard"`
	Easy []string `json:"easy"`
}

func NewAhoComplexityKeywordMatcher(mappingPath string) (ComplexityKeywordMatcher, error) {
	resolvedPath := config.ResolveModelPath(mappingPath)
	if strings.TrimSpace(resolvedPath) == "" {
		resolvedPath = mappingPath
	}

	mapping, err := loadComplexityKeywordMapping(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load complexity keyword mapping: %w", err)
	}
	if len(mapping.Hard) == 0 && len(mapping.Easy) == 0 {
		return nil, fmt.Errorf("complexity keyword mapping has no valid hard/easy keywords")
	}

	classifier := nlp_binding.NewAhoClassifier()
	if len(mapping.Hard) > 0 {
		if err := classifier.AddRule("hard", mapping.Hard, false); err != nil {
			return nil, fmt.Errorf("failed to add complexity hard keywords: %w", err)
		}
	}
	if len(mapping.Easy) > 0 {
		if err := classifier.AddRule("easy", mapping.Easy, false); err != nil {
			return nil, fmt.Errorf("failed to add complexity easy keywords: %w", err)
		}
	}

	return &AhoComplexityKeywordMatcher{classifier: classifier}, nil
}

func (m *AhoComplexityKeywordMatcher) Classify(text string) (string, bool, error) {
	if m == nil || m.classifier == nil {
		return "", false, fmt.Errorf("complexity keyword matcher is not initialized")
	}

	result := m.classifier.Classify(text)
	level := strings.ToLower(strings.TrimSpace(result.RuleName))
	if !result.Matched || (level != "hard" && level != "easy") {
		return "", false, nil
	}
	return level, true, nil
}

func (m *AhoComplexityKeywordMatcher) Free() {
	if m != nil && m.classifier != nil {
		m.classifier.Free()
	}
}

func loadComplexityKeywordMapping(path string) (complexityKeywordMapping, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return complexityKeywordMapping{}, fmt.Errorf("read %q failed: %w", path, err)
	}

	var mapping complexityKeywordMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return complexityKeywordMapping{}, fmt.Errorf("unsupported complexity keyword mapping format: %w", err)
	}

	mapping.Hard = normalizeKeywordList(mapping.Hard)
	mapping.Easy = normalizeKeywordList(mapping.Easy)
	return mapping, nil
}

func normalizeKeywordList(input []string) []string {
	seen := make(map[string]struct{})
	output := make([]string, 0, len(input))
	for _, kw := range input {
		trimmed := strings.TrimSpace(kw)
		if trimmed == "" {
			continue
		}
		normalized := strings.ToLower(trimmed)
		if _, exists := seen[normalized]; exists {
			continue
		}
		seen[normalized] = struct{}{}
		output = append(output, trimmed)
	}
	return output
}
