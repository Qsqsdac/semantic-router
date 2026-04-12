package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFactCheckSVMClassifier_Classify(t *testing.T) {
	classifier, err := NewFactCheckSVMClassifier()
	if err != nil {
		t.Fatalf("failed to create svm classifier: %v", err)
	}

	result, err := classifier.Classify("Who is the first president of America?")
	if err != nil {
		t.Fatalf("svm classify failed: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Label != FactCheckLabelNeeded && result.Label != FactCheckLabelNotNeeded {
		t.Fatalf("unexpected label: %s", result.Label)
	}
	if result.Confidence <= 0 || result.Confidence > 1 {
		t.Fatalf("confidence out of range: %f", result.Confidence)
	}
}

func TestFactCheckClassifier_SVMOnlyInitializeWithoutBERTModel(t *testing.T) {
	cfg := &config.FactCheckModelConfig{
		Mode:      config.FactCheckModeSVMOnly,
		Threshold: 0.7,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("failed to create classifier: %v", err)
	}
	if err := classifier.Initialize(); err != nil {
		t.Fatalf("failed to initialize svm_only classifier: %v", err)
	}

	result, err := classifier.Classify("Please summarize this paragraph.")
	if err != nil {
		t.Fatalf("svm_only classify failed: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestFactCheckClassifier_SVMFallbackRequiresBERTModelID(t *testing.T) {
	cfg := &config.FactCheckModelConfig{
		Mode:      config.FactCheckModeSVMFallbackBERT,
		Threshold: 0.7,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("failed to create classifier: %v", err)
	}
	if err := classifier.Initialize(); err == nil {
		t.Fatal("expected initialization error when svm_fallback_bert has no model_id")
	}
}
