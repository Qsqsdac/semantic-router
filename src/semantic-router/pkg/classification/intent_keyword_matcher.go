package classification

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type IntentKeywordMatcher interface {
	Classify(text string) (string, bool, error)
	Free()
}

type AhoIntentKeywordMatcher struct {
	classifier *nlp_binding.AhoClassifier
}

type intentKeywordRule struct {
	Category string   `json:"category"`
	Keywords []string `json:"keywords"`
}

type intentKeywordRuleList struct {
	Rules []intentKeywordRule `json:"rules"`
}

func NewAhoIntentKeywordMatcher(mappingPath string, caseSensitive bool) (IntentKeywordMatcher, error) {
	resolvedPath := config.ResolveModelPath(mappingPath)
	if strings.TrimSpace(resolvedPath) == "" {
		resolvedPath = mappingPath
	}

	rules, err := loadIntentKeywordRules(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load intent keyword mapping: %w", err)
	}
	if len(rules) == 0 {
		return nil, fmt.Errorf("intent keyword mapping has no valid category rules")
	}

	classifier := nlp_binding.NewAhoClassifier()
	for _, rule := range rules {
		if err := classifier.AddRule(rule.Category, rule.Keywords, caseSensitive); err != nil {
			return nil, fmt.Errorf("failed to add intent keyword rule %q: %w", rule.Category, err)
		}
	}

	return &AhoIntentKeywordMatcher{classifier: classifier}, nil
}

func (m *AhoIntentKeywordMatcher) Classify(text string) (string, bool, error) {
	if m == nil || m.classifier == nil {
		return "", false, fmt.Errorf("intent keyword matcher is not initialized")
	}

	result := m.classifier.Classify(text)
	if !result.Matched || strings.TrimSpace(result.RuleName) == "" {
		return "", false, nil
	}
	return result.RuleName, true, nil
}

func (m *AhoIntentKeywordMatcher) Free() {
	if m != nil && m.classifier != nil {
		m.classifier.Free()
	}
}

func loadIntentKeywordRules(path string) ([]intentKeywordRule, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %q failed: %w", path, err)
	}

	var ruleList intentKeywordRuleList
	if err := json.Unmarshal(data, &ruleList); err == nil && len(ruleList.Rules) > 0 {
		return normalizeIntentKeywordRules(ruleList.Rules), nil
	}

	var byCategory map[string][]string
	if err := json.Unmarshal(data, &byCategory); err != nil {
		return nil, fmt.Errorf("unsupported intent keyword mapping format: %w", err)
	}

	categories := make([]string, 0, len(byCategory))
	for category := range byCategory {
		categories = append(categories, category)
	}
	sort.Strings(categories)

	rules := make([]intentKeywordRule, 0, len(categories))
	for _, category := range categories {
		rules = append(rules, intentKeywordRule{
			Category: category,
			Keywords: byCategory[category],
		})
	}
	return normalizeIntentKeywordRules(rules), nil
}

func normalizeIntentKeywordRules(input []intentKeywordRule) []intentKeywordRule {
	normalized := make([]intentKeywordRule, 0, len(input))
	for _, rule := range input {
		category := strings.TrimSpace(rule.Category)
		if category == "" {
			continue
		}

		seen := make(map[string]bool)
		keywords := make([]string, 0, len(rule.Keywords))
		for _, kw := range rule.Keywords {
			trimmed := strings.TrimSpace(kw)
			if trimmed == "" || seen[trimmed] {
				continue
			}
			seen[trimmed] = true
			keywords = append(keywords, trimmed)
		}
		if len(keywords) == 0 {
			continue
		}

		normalized = append(normalized, intentKeywordRule{
			Category: category,
			Keywords: keywords,
		})
	}
	return normalized
}
