package classification

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadIntentKeywordRulesWithCategoryMapFormat(t *testing.T) {
	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "intent_keywords.json")
	data := []byte(`{
  "economics": ["stock market", "inflation"],
  "health": ["symptom", "diagnosis"]
}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write test mapping failed: %v", err)
	}

	rules, err := loadIntentKeywordRules(path)
	if err != nil {
		t.Fatalf("loadIntentKeywordRules returned error: %v", err)
	}
	if len(rules) != 2 {
		t.Fatalf("expected 2 rules, got %d", len(rules))
	}
}

func TestLoadIntentKeywordRulesWithRulesListFormat(t *testing.T) {
	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "intent_keywords_rules.json")
	data := []byte(`{
  "rules": [
    {"category": "economics", "keywords": ["stock market", "stock market", "inflation"]},
    {"category": "", "keywords": ["ignored"]}
  ]
}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write test mapping failed: %v", err)
	}

	rules, err := loadIntentKeywordRules(path)
	if err != nil {
		t.Fatalf("loadIntentKeywordRules returned error: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 normalized rule, got %d", len(rules))
	}
	if got := len(rules[0].Keywords); got != 2 {
		t.Fatalf("expected duplicate keywords to be removed, got %d keywords", got)
	}
}
