package classification

import (
	"fmt"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FactCheckResult represents the result of fact-check classification
type FactCheckResult struct {
	NeedsFactCheck bool    `json:"needs_fact_check"`
	Confidence     float32 `json:"confidence"`
	Label          string  `json:"label"` // "FACT_CHECK_NEEDED" or "NO_FACT_CHECK_NEEDED"
}

// FactCheckClassifier handles fact-check classification to determine if a prompt
// requires external factual verification using the halugate-sentinel ML model
type FactCheckClassifier struct {
	config        *config.FactCheckModelConfig
	mapping       *FactCheckMapping
	initialized   bool
	useMmBERT32K  bool // Track if mmBERT-32K is used for inference
	mode          string
	svmClassifier *FactCheckSVMClassifier
	mu            sync.RWMutex
}

// NewFactCheckClassifier creates a new fact-check classifier
func NewFactCheckClassifier(cfg *config.FactCheckModelConfig) (*FactCheckClassifier, error) {
	if cfg == nil {
		return nil, nil // Disabled
	}

	classifier := &FactCheckClassifier{
		config: cfg,
	}

	return classifier, nil
}

// Initialize initializes the fact-check classifier with the halugate-sentinel ML model
func (c *FactCheckClassifier) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil
	}

	// Use default mapping (no external mapping file needed)
	c.mapping = &FactCheckMapping{
		LabelToIdx: map[string]int{
			FactCheckLabelNotNeeded: 0,
			FactCheckLabelNeeded:    1,
		},
		IdxToLabel: map[string]string{
			"0": FactCheckLabelNotNeeded,
			"1": FactCheckLabelNeeded,
		},
	}

	// Initialize ML model - ModelID is required
	c.mode = c.config.EffectiveMode()

	logging.Infof("Initializing Fact-Check Classifier:")
	logging.Infof("Mode: %s", c.mode)

	switch c.mode {
	case config.FactCheckModeSVMOnly:
		svmClassifier, err := NewFactCheckSVMClassifier()
		if err != nil {
			return fmt.Errorf("failed to initialize fact-check svm classifier: %w", err)
		}
		c.svmClassifier = svmClassifier
		c.initialized = true
		logging.Infof("Fact-check classifier initialized successfully (svm_only)")
		return nil

	case config.FactCheckModeSVMFallbackBERT:
		svmClassifier, err := NewFactCheckSVMClassifier()
		if err != nil {
			return fmt.Errorf("failed to initialize fact-check svm classifier: %w", err)
		}
		c.svmClassifier = svmClassifier

		if c.config.ModelID == "" {
			return fmt.Errorf("fact-check mode %q requires model_id for bert fallback", config.FactCheckModeSVMFallbackBERT)
		}
		if err := c.initializeBERTLocked(); err != nil {
			return err
		}

		c.initialized = true
		logging.Infof("Fact-check classifier initialized successfully (svm_fallback_bert)")
		return nil

	default:
		if c.config.ModelID == "" {
			return fmt.Errorf("fact-check classifier requires model_id to be configured in bert mode")
		}
		if err := c.initializeBERTLocked(); err != nil {
			return err
		}
		c.initialized = true
		logging.Infof("Fact-check classifier initialized successfully (bert)")
		return nil
	}
}

func (c *FactCheckClassifier) initializeBERTLocked() error {
	logging.Infof("Model: %s", c.config.ModelID)
	logging.Infof("CPU Mode: %v", c.config.UseCPU)

	if c.config.UseMmBERT32K {
		logging.Infof("Type: mmBERT-32K (32K context, YaRN RoPE)")
		err := candle.InitMmBert32KFactcheckClassifier(c.config.ModelID, c.config.UseCPU)
		if err != nil {
			return fmt.Errorf("failed to initialize mmBERT-32K fact-check model from %s: %w", c.config.ModelID, err)
		}
		c.useMmBERT32K = true
		return nil
	}

	logging.Infof("Type: halugate-sentinel (ML-based)")
	err := candle.InitFactCheckClassifier(c.config.ModelID, c.config.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize fact-check ML model from %s: %w", c.config.ModelID, err)
	}
	c.useMmBERT32K = false
	return nil
}

func (c *FactCheckClassifier) classifyWithBERT(text string) (*FactCheckResult, error) {
	var result candle.ClassResult
	var err error
	if c.useMmBERT32K {
		result, err = candle.ClassifyMmBert32KFactcheck(text)
	} else {
		result, err = candle.ClassifyFactCheckText(text)
	}
	if err != nil {
		return nil, fmt.Errorf("fact-check ML classification failed: %w", err)
	}

	needsFactCheck := result.Class == 1
	confidence := result.Confidence

	label := FactCheckLabelNotNeeded
	if needsFactCheck {
		label = FactCheckLabelNeeded
	}

	threshold := c.config.Threshold
	if threshold <= 0 {
		threshold = 0.7
	}

	if needsFactCheck && confidence < threshold {
		needsFactCheck = false
		label = FactCheckLabelNotNeeded
		confidence = 1.0 - confidence
	}

	return &FactCheckResult{
		NeedsFactCheck: needsFactCheck,
		Confidence:     confidence,
		Label:          label,
	}, nil
}

func (c *FactCheckClassifier) svmFallbackThreshold() float32 {
	threshold := c.config.SVMFallbackThreshold
	if threshold <= 0 {
		threshold = c.config.Threshold
	}
	if threshold <= 0 {
		threshold = 0.7
	}
	return threshold
}

// Classify determines if a prompt needs fact-checking using the ML model
func (c *FactCheckClassifier) Classify(text string) (*FactCheckResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("fact-check classifier not initialized")
	}

	if text == "" {
		return &FactCheckResult{
			NeedsFactCheck: false,
			Confidence:     1.0,
			Label:          FactCheckLabelNotNeeded,
		}, nil
	}

	if c.mode == config.FactCheckModeSVMOnly {
		if c.svmClassifier == nil {
			return nil, fmt.Errorf("fact-check svm classifier is not initialized")
		}
		result, err := c.svmClassifier.Classify(text)
		if err != nil {
			return nil, fmt.Errorf("fact-check svm classification failed: %w", err)
		}
		logging.Debugf("Fact-check SVM classification: text_len=%d, needs_fact_check=%v, confidence=%.3f", len(text), result.NeedsFactCheck, result.Confidence)
		return result, nil
	}

	if c.mode == config.FactCheckModeSVMFallbackBERT {
		if c.svmClassifier != nil {
			result, err := c.svmClassifier.Classify(text)
			if err == nil {
				if result.Confidence >= c.svmFallbackThreshold() {
					logging.Debugf("Fact-check SVM classification: text_len=%d, needs_fact_check=%v, confidence=%.3f", len(text), result.NeedsFactCheck, result.Confidence)
					return result, nil
				}
				logging.Infof("Fact-check SVM confidence %.3f below fallback threshold %.3f, fallback to BERT", result.Confidence, c.svmFallbackThreshold())
			} else {
				logging.Warnf("Fact-check SVM classification failed, fallback to BERT: %v", err)
			}
		}
		result, err := c.classifyWithBERT(text)
		if err != nil {
			return nil, err
		}
		logging.Debugf("Fact-check BERT fallback classification: text_len=%d, needs_fact_check=%v, confidence=%.3f", len(text), result.NeedsFactCheck, result.Confidence)
		return result, nil
	}

	result, err := c.classifyWithBERT(text)
	if err != nil {
		return nil, err
	}
	logging.Debugf("Fact-check ML classification: text_len=%d, needs_fact_check=%v, confidence=%.3f", len(text), result.NeedsFactCheck, result.Confidence)
	return result, nil
}

// IsInitialized returns whether the classifier is initialized
func (c *FactCheckClassifier) IsInitialized() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.initialized
}

// GetMapping returns the fact-check mapping
func (c *FactCheckClassifier) GetMapping() *FactCheckMapping {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.mapping
}
