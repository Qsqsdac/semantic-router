package classification

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
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

	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr bytes.Buffer
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

	classifier := &FastTextIntentClassifier{
		binaryPath: binaryPath,
		modelPath:  modelPath,
		threshold:  threshold,
		timeout:    timeout,
	}

	if err := classifier.initializePersistentProcess(); err != nil {
		return nil, err
	}

	logging.Infof("Initialized persistent fastText intent process (threshold=%.3f, timeout=%s)", threshold, timeout)
	return classifier, nil
}

func (f *FastTextIntentClassifier) initializePersistentProcess() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if err := f.startProcessLocked(); err != nil {
		return err
	}

	probe, err := f.requestPredictionLineLocked("startup health check")
	if err != nil {
		f.stopProcessLocked()
		return err
	}
	if strings.TrimSpace(probe) == "" {
		f.stopProcessLocked()
		return fmt.Errorf("fastText startup probe returned empty output")
	}

	return nil
}

func (f *FastTextIntentClassifier) startProcessLocked() error {
	f.stderr.Reset()

	cmd := exec.Command(f.binaryPath, "predict-prob", f.modelPath, "-", "1")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to open stdin for fastText: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return fmt.Errorf("failed to open stdout for fastText: %w", err)
	}
	cmd.Stderr = &f.stderr

	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return fmt.Errorf("failed to start fastText: %w", err)
	}

	f.cmd = cmd
	f.stdin = stdin
	f.stdout = bufio.NewReader(stdout)
	return nil
}

func (f *FastTextIntentClassifier) stopProcessLocked() {
	if f.stdin != nil {
		_ = f.stdin.Close()
	}
	if f.cmd != nil && f.cmd.Process != nil {
		_ = f.cmd.Process.Kill()
		_ = f.cmd.Wait()
	}
	f.cmd = nil
	f.stdin = nil
	f.stdout = nil
}

func (f *FastTextIntentClassifier) ensureProcessLocked() error {
	if f.cmd != nil && f.cmd.ProcessState == nil {
		return nil
	}
	f.stopProcessLocked()
	if err := f.startProcessLocked(); err != nil {
		return err
	}
	return nil
}

func (f *FastTextIntentClassifier) requestPredictionLineLocked(text string) (string, error) {
	cleaned := strings.ReplaceAll(text, "\n", " ") + "\n"
	if _, err := io.WriteString(f.stdin, cleaned); err != nil {
		return "", fmt.Errorf("failed to write to fastText stdin: %w", err)
	}

	type readResult struct {
		line string
		err  error
	}
	resultCh := make(chan readResult, 1)
	go func() {
		line, err := f.stdout.ReadString('\n')
		resultCh <- readResult{line: line, err: err}
	}()

	select {
	case r := <-resultCh:
		if r.err != nil {
			return "", fmt.Errorf("failed to read fastText output: %w", r.err)
		}
		return strings.TrimSpace(r.line), nil
	case <-time.After(f.timeout):
		f.stopProcessLocked()
		return "", fmt.Errorf("fastText prediction timed out after %s", f.timeout)
	}
}

// Classify predicts the best-matching intent using a persistent FastText process.
func (f *FastTextIntentClassifier) Classify(text string) (string, float64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if err := f.ensureProcessLocked(); err != nil {
		return "", 0, fmt.Errorf("failed to ensure fastText process: %w", err)
	}

	output, err := f.requestPredictionLineLocked(text)
	if err != nil {
		return "", 0, fmt.Errorf("fastText prediction failed: %w: %s", err, strings.TrimSpace(f.stderr.String()))
	}
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
