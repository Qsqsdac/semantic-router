package classification

import (
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/factchecksvm"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lineartoken"
)

// FactCheckSVMClassifier runs AC + linear SVM inference in pure Go.
type FactCheckSVMClassifier struct {
	classifier *lineartoken.Classifier
}

func NewFactCheckSVMClassifier() (*FactCheckSVMClassifier, error) {
	classifier, err := lineartoken.New(factchecksvm.Model())
	if err != nil {
		return nil, fmt.Errorf("initialize fact-check linear token model: %w", err)
	}
	return &FactCheckSVMClassifier{classifier: classifier}, nil
}

func (c *FactCheckSVMClassifier) Classify(text string) (*FactCheckResult, error) {
	if c == nil {
		return nil, fmt.Errorf("svm classifier is nil")
	}

	label, positive, probability := c.classifier.Predict(text)
	needsFactCheck := label == "needs_fact_check"
	resolvedLabel, resolvedNeed := normalizeFactCheckLabel(label, needsFactCheck)

	return &FactCheckResult{
		NeedsFactCheck: resolvedNeed,
		Confidence:     probabilityToConfidence(positive, probability),
		Label:          resolvedLabel,
	}, nil
}

func probabilityToConfidence(positive bool, probability float64) float32 {
	if !positive {
		probability = 1 - probability
	}
	return float32(math.Max(probability, 1-probability))
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
