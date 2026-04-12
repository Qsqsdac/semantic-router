package classification

import (
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/factchecksvm"
)

// FactCheckSVMClassifier runs AC + linear SVM inference in pure Go.
type FactCheckSVMClassifier struct{}

func NewFactCheckSVMClassifier() (*FactCheckSVMClassifier, error) {
	return &FactCheckSVMClassifier{}, nil
}

func (c *FactCheckSVMClassifier) Classify(text string) (*FactCheckResult, error) {
	if c == nil {
		return nil, fmt.Errorf("svm classifier is nil")
	}

	label, needsFactCheck, score := factchecksvm.Predict(text)
	resolvedLabel, resolvedNeed := normalizeFactCheckLabel(label, needsFactCheck)

	return &FactCheckResult{
		NeedsFactCheck: resolvedNeed,
		Confidence:     marginToConfidence(score),
		Label:          resolvedLabel,
	}, nil
}

func marginToConfidence(score float64) float32 {
	margin := math.Abs(score)
	return float32(1.0 / (1.0 + math.Exp(-margin)))
}

func normalizeFactCheckLabel(label string, needsFactCheck bool) (string, bool) {
	canon := strings.ToLower(strings.TrimSpace(label))
	switch canon {
	case "fact_check_needed", "needs_fact_check", "fact check needed":
		return FactCheckLabelNeeded, true
	case "no_fact_check_needed", "no fact check needed":
		return FactCheckLabelNotNeeded, false
	}
	if needsFactCheck {
		return FactCheckLabelNeeded, true
	}
	return FactCheckLabelNotNeeded, false
}
