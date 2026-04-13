package classification

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadComplexityKeywordMapping(t *testing.T) {
	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "complexity_keywords.json")
	data := []byte(`{
  "hard": ["step by step", "tradeoff", "step by step"],
  "easy": ["quick summary", "  "]
}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write test mapping failed: %v", err)
	}

	mapping, err := loadComplexityKeywordMapping(path)
	if err != nil {
		t.Fatalf("loadComplexityKeywordMapping returned error: %v", err)
	}
	if got := len(mapping.Hard); got != 2 {
		t.Fatalf("expected 2 unique hard keywords, got %d", got)
	}
	if got := len(mapping.Easy); got != 1 {
		t.Fatalf("expected 1 valid easy keyword, got %d", got)
	}
}
