package classification

import (
	"fmt"
	"strings"
	"time"

	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultFastTextThreshold = 0.5
	defaultFastTextTimeout   = 2 * time.Second
)

// IntentFastTextClassifier defines the contract for FastText-based intent classification.
type IntentFastTextClassifier interface {
	// Classify returns the predicted category, probability, and any error encountered.
	// If probability is below threshold, category should be empty.
	Classify(text string) (string, float64, error)
}

// FastTextIntentClassifier runs fastText inference in-process through nlp-binding.
type FastTextIntentClassifier struct {
	threshold float64
	model     *nlp_binding.FastTextClassifier
}

// NewFastTextIntentClassifier builds an in-process CPU fastText classifier with sane defaults.
// binaryPath and timeout are accepted for backward-compatible config parsing.
func NewFastTextIntentClassifier(binaryPath, modelPath string, threshold float64, timeout time.Duration) (IntentFastTextClassifier, error) {
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("intent_fasttext_model_path is required for fastText mode")
	}
	if threshold <= 0 || threshold > 1 {
		threshold = defaultFastTextThreshold
	}
	if strings.TrimSpace(binaryPath) != "" {
		logging.Infof("intent_fasttext_binary_path is ignored by in-process fastText inference")
	}
	if timeout > 0 {
		logging.Infof("intent_fasttext_timeout_seconds is ignored by in-process fastText inference")
	}

	model, err := nlp_binding.NewFastTextClassifier(modelPath)
	if err != nil {
		return nil, err
	}

	classifier := &FastTextIntentClassifier{
		threshold: threshold,
		model:     model,
	}

	logging.Infof("Initialized in-process fastText intent model (threshold=%.3f)", threshold)
	return classifier, nil
}

// Classify predicts the best-matching intent using the in-process fastText model.
func (f *FastTextIntentClassifier) Classify(text string) (string, float64, error) {
	cleaned := strings.ReplaceAll(text, "\n", " ")
	if strings.TrimSpace(cleaned) == "" {
		return "", 0, nil
	}
	if f.model == nil {
		return "", 0, fmt.Errorf("fastText model is not initialized")
	}

	prediction, err := f.model.Predict(cleaned, 0)
	if err != nil {
		return "", 0, err
	}
	if !prediction.Matched {
		return "", 0, nil
	}

	label := strings.TrimPrefix(prediction.Label, "__label__")
	prob := float64(prediction.Probability)

	if prob < f.threshold {
		logging.Infof("fastText probability %.3f below threshold %.3f", prob, f.threshold)
		return "", prob, nil
	}

	return label, prob, nil
}
