package classification

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"time"

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

// FastTextIntentClassifier runs the FastText CLI as a subprocess for inference.
type FastTextIntentClassifier struct {
	binaryPath string
	modelPath  string
	threshold  float64
	timeout    time.Duration
}

// NewFastTextIntentClassifier builds a CLI-backed FastText classifier with sane defaults.
func NewFastTextIntentClassifier(binaryPath, modelPath string, threshold float64, timeout time.Duration) (IntentFastTextClassifier, error) {
	if strings.TrimSpace(modelPath) == "" {
		return nil, fmt.Errorf("intent_fasttext_model_path is required for fastText mode")
	}
	if strings.TrimSpace(binaryPath) == "" {
		binaryPath = "fasttext"
	}
	if threshold <= 0 || threshold > 1 {
		threshold = defaultFastTextThreshold
	}
	if timeout <= 0 {
		timeout = defaultFastTextTimeout
	}
	return &FastTextIntentClassifier{
		binaryPath: binaryPath,
		modelPath:  modelPath,
		threshold:  threshold,
		timeout:    timeout,
	}, nil
}

// Classify predicts the best-matching intent using FastText `predict-prob`.
func (f *FastTextIntentClassifier) Classify(text string) (string, float64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), f.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, f.binaryPath, "predict-prob", f.modelPath, "-", "1")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", 0, fmt.Errorf("failed to open stdin for fastText: %w", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return "", 0, fmt.Errorf("failed to start fastText: %w", err)
	}

	cleaned := strings.ReplaceAll(text, "\n", " ") + "\n"
	if _, err := io.WriteString(stdin, cleaned); err != nil {
		return "", 0, fmt.Errorf("failed to write to fastText stdin: %w", err)
	}
	_ = stdin.Close()

	if err := cmd.Wait(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return "", 0, fmt.Errorf("fastText prediction timed out after %s", f.timeout)
		}
		return "", 0, fmt.Errorf("fastText prediction failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	output := strings.TrimSpace(stdout.String())
	if output == "" {
		return "", 0, nil
	}

	fields := strings.Fields(output)
	if len(fields) < 2 {
		return "", 0, fmt.Errorf("unexpected fastText output: %q", output)
	}

	label := strings.TrimPrefix(fields[0], "__label__")
	prob, err := strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return "", 0, fmt.Errorf("invalid fastText probability %q: %w", fields[1], err)
	}

	if prob < f.threshold {
		logging.Infof("fastText probability %.3f below threshold %.3f", prob, f.threshold)
		return "", prob, nil
	}

	return label, prob, nil
}
