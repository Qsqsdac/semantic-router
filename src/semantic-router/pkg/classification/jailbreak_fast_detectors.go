package classification

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/jailbreaklinearsvm"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lineartoken"
)

// MinHashJailbreakDetector is a CPU-only approximate matcher for known attack
// templates. It is intentionally a high-precision gate; misses continue to L2.
type MinHashJailbreakDetector struct {
	shingleSize   int
	seeds         []uint64
	patterns      []minHashPattern
	patternIndex  map[string][]minHashPatternAnchor
	patternSizes  map[int]struct{}
	exactPatterns map[int]map[uint64][]int
	regexRules    []jailbreakRegexRule
	threshold     float32
}

type minHashPattern struct {
	signature []uint64
	tokens    []string
	rares     []string
}

type minHashPatternAnchor struct {
	patternIndex int
	tokenOffset  int
}

type jailbreakRegexRule struct {
	anchor  string
	pattern *regexp.Regexp
}

type jailbreakRegexModel struct {
	Rules []jailbreakRegexRuleConfig `json:"rules"`
}

type jailbreakRegexRuleConfig struct {
	Anchor  string `json:"anchor"`
	Pattern string `json:"pattern"`
}

// BenignFastDetector is a conservative allow gate. Every allow rule must
// match a narrow task shape, and any configured risk term vetoes the result.
type BenignFastDetector struct {
	riskTerms []string
	rules     []benignFastRule
}

type benignFastRule struct {
	anchor  string
	pattern *regexp.Regexp
}

type benignFastModel struct {
	RiskTerms []string               `json:"risk_terms"`
	Rules     []benignFastRuleConfig `json:"rules"`
}

type benignFastRuleConfig struct {
	Anchor  string `json:"anchor"`
	Pattern string `json:"pattern"`
}

func NewMinHashJailbreakDetector(rules []string, shingleSize, permutations int, threshold float32) *MinHashJailbreakDetector {
	if shingleSize < 1 {
		shingleSize = 3
	}
	if permutations < 8 {
		permutations = 64
	}
	if threshold <= 0 || threshold > 1 {
		threshold = 0.92
	}
	d := &MinHashJailbreakDetector{
		shingleSize:   shingleSize,
		patternIndex:  make(map[string][]minHashPatternAnchor),
		patternSizes:  make(map[int]struct{}),
		exactPatterns: make(map[int]map[uint64][]int),
		threshold:     threshold,
	}
	for i := 0; i < permutations; i++ {
		d.seeds = append(d.seeds, uint64(i+1)*0x9e3779b185ebca87)
	}
	for _, pattern := range rules {
		tokens := tokenizeJailbreakText(pattern)
		if signature := d.signatureTokens(tokens, 0, len(tokens)); len(signature) > 0 {
			patternIndex := len(d.patterns)
			rareTokens := d.selectRareTokens(tokens, rules)
			d.patterns = append(d.patterns, minHashPattern{
				signature: signature,
				tokens:    tokens,
				rares:     rareTokens,
			})
			d.patternSizes[len(tokens)] = struct{}{}
			if d.exactPatterns[len(tokens)] == nil {
				d.exactPatterns[len(tokens)] = make(map[uint64][]int)
			}
			d.exactPatterns[len(tokens)][tokenSequenceHash(tokens, 0, len(tokens))] = append(
				d.exactPatterns[len(tokens)][tokenSequenceHash(tokens, 0, len(tokens))], len(d.patterns)-1)
			for tokenOffset, token := range tokens {
				if containsJailbreakToken(rareTokens, token) {
					d.patternIndex[token] = append(d.patternIndex[token], minHashPatternAnchor{
						patternIndex: patternIndex,
						tokenOffset:  tokenOffset,
					})
				}
			}
		}
	}
	return d
}

func (d *MinHashJailbreakDetector) selectRareTokens(tokens []string, rules []string) []string {
	frequencies := make(map[string]int)
	for _, rule := range rules {
		seen := make(map[string]struct{})
		for _, token := range tokenizeJailbreakText(rule) {
			if _, ok := seen[token]; !ok {
				frequencies[token]++
				seen[token] = struct{}{}
			}
		}
	}
	if len(tokens) == 0 {
		return nil
	}
	minFrequency := math.MaxInt
	for _, token := range tokens {
		if frequencies[token] < minFrequency {
			minFrequency = frequencies[token]
		}
	}
	var rares []string
	for _, token := range tokens {
		if frequencies[token] == minFrequency {
			rares = append(rares, token)
		}
	}
	return rares
}

func containsJailbreakToken(tokens []string, target string) bool {
	for _, token := range tokens {
		if token == target {
			return true
		}
	}
	return false
}

func LoadMinHashPatterns(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read minhash model: %w", err)
	}
	var patterns []string
	if err := json.Unmarshal(data, &patterns); err == nil {
		return patterns, nil
	}
	var catalog struct {
		Patterns []string `json:"patterns"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		return nil, fmt.Errorf("parse minhash model: %w", err)
	}
	return catalog.Patterns, nil
}

func LoadJailbreakRegexRules(path string) ([]jailbreakRegexRule, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read jailbreak regex model: %w", err)
	}
	var model jailbreakRegexModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("parse jailbreak regex model: %w", err)
	}
	rules := make([]jailbreakRegexRule, 0, len(model.Rules))
	for index, rule := range model.Rules {
		anchor := strings.ToLower(strings.TrimSpace(rule.Anchor))
		if anchor == "" || strings.TrimSpace(rule.Pattern) == "" {
			return nil, fmt.Errorf("jailbreak regex rule %d requires anchor and pattern", index)
		}
		compiled, err := regexp.Compile(rule.Pattern)
		if err != nil {
			return nil, fmt.Errorf("compile jailbreak regex rule %d: %w", index, err)
		}
		rules = append(rules, jailbreakRegexRule{anchor: anchor, pattern: compiled})
	}
	return rules, nil
}

func LoadBenignFastDetector(path string) (*BenignFastDetector, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read benign fast model: %w", err)
	}
	var model benignFastModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("parse benign fast model: %w", err)
	}
	detector := &BenignFastDetector{}
	for _, term := range model.RiskTerms {
		term = strings.ToLower(strings.TrimSpace(term))
		if term != "" {
			detector.riskTerms = append(detector.riskTerms, term)
		}
	}
	for index, rule := range model.Rules {
		anchor := strings.ToLower(strings.TrimSpace(rule.Anchor))
		if anchor == "" || strings.TrimSpace(rule.Pattern) == "" {
			return nil, fmt.Errorf("benign fast rule %d requires anchor and pattern", index)
		}
		compiled, err := regexp.Compile(rule.Pattern)
		if err != nil {
			return nil, fmt.Errorf("compile benign fast rule %d: %w", index, err)
		}
		detector.rules = append(detector.rules, benignFastRule{anchor: anchor, pattern: compiled})
	}
	if len(detector.rules) == 0 {
		return nil, fmt.Errorf("benign fast model requires at least one rule")
	}
	return detector, nil
}

func (d *BenignFastDetector) Match(text string) bool {
	lowerText := strings.ToLower(text)
	for _, term := range d.riskTerms {
		if strings.Contains(lowerText, term) {
			return false
		}
	}
	for _, rule := range d.rules {
		if strings.Contains(lowerText, rule.anchor) && rule.pattern.MatchString(text) {
			return true
		}
	}
	return false
}

func (d *MinHashJailbreakDetector) SetRegexRules(rules []jailbreakRegexRule) {
	d.regexRules = rules
}

func (d *MinHashJailbreakDetector) MatchRegex(text string) bool {
	lowerText := strings.ToLower(text)
	for _, rule := range d.regexRules {
		if strings.Contains(lowerText, rule.anchor) && rule.pattern.MatchString(text) {
			return true
		}
	}
	return false
}

func (d *MinHashJailbreakDetector) Match(text string) (bool, float32) {
	if len(d.patterns) == 0 {
		return false, 0
	}
	tokens := tokenizeJailbreakText(text)
	if len(tokens) == 0 {
		return false, 0
	}
	tokenHashes := make([]uint64, len(tokens))
	for i, token := range tokens {
		tokenHashes[i] = hashJailbreakToken(token)
	}
	for patternSize, patterns := range d.exactPatterns {
		if len(tokens) < patternSize {
			continue
		}
		for start := 0; start+patternSize <= len(tokens); start++ {
			if patternIndexes, ok := patterns[tokenSequenceHashFromHashes(tokenHashes, start, patternSize)]; ok {
				for _, patternIndex := range patternIndexes {
					if equalJailbreakTokens(tokens, start, d.patterns[patternIndex].tokens) {
						return true, 1
					}
				}
			}
		}
	}
	candidateWindows := make(map[int]map[int]struct{})
	for tokenIndex, token := range tokens {
		for _, anchor := range d.patternIndex[token] {
			windowStart := tokenIndex - anchor.tokenOffset
			patternSize := len(d.patterns[anchor.patternIndex].tokens)
			if windowStart < 0 || windowStart+patternSize > len(tokens) {
				continue
			}
			if candidateWindows[anchor.patternIndex] == nil {
				candidateWindows[anchor.patternIndex] = make(map[int]struct{})
			}
			candidateWindows[anchor.patternIndex][windowStart] = struct{}{}
		}
	}
	if len(candidateWindows) == 0 {
		return false, 0
	}
	best := float32(0)
	for patternIndex, starts := range candidateWindows {
		pattern := d.patterns[patternIndex]
		for start := range starts {
			signature := d.signatureTokenHashes(tokenHashes, start, len(pattern.tokens))
			equal := 0
			for i := range signature {
				if signature[i] == pattern.signature[i] {
					equal++
				}
			}
			score := float32(equal) / float32(len(signature))
			if score > best {
				best = score
			}
		}
	}
	return best >= d.threshold, best
}

func tokenizeJailbreakText(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return unicode.IsSpace(r) || unicode.IsPunct(r) })
}

func tokenSequenceHash(tokens []string, start, length int) uint64 {
	result := uint64(0xcbf29ce484222325)
	for _, token := range tokens[start : start+length] {
		result ^= hashJailbreakToken(token)
		result *= 0x100000001b3
	}
	return result
}

func tokenSequenceHashFromHashes(tokenHashes []uint64, start, length int) uint64 {
	result := uint64(0xcbf29ce484222325)
	for _, tokenHash := range tokenHashes[start : start+length] {
		result ^= tokenHash
		result *= 0x100000001b3
	}
	return result
}

func equalJailbreakTokens(tokens []string, start int, pattern []string) bool {
	if start < 0 || start+len(pattern) > len(tokens) {
		return false
	}
	for i, token := range pattern {
		if tokens[start+i] != token {
			return false
		}
	}
	return true
}

func (d *MinHashJailbreakDetector) signature(text string) []uint64 {
	tokens := tokenizeJailbreakText(text)
	return d.signatureTokens(tokens, 0, len(tokens))
}

func (d *MinHashJailbreakDetector) signatureTokens(tokens []string, start, length int) []uint64 {
	if length < d.shingleSize || start < 0 || start+length > len(tokens) {
		return nil
	}
	tokenHashes := make([]uint64, len(tokens))
	for i, token := range tokens {
		tokenHashes[i] = hashJailbreakToken(token)
	}
	return d.signatureTokenHashes(tokenHashes, start, length)
}

func (d *MinHashJailbreakDetector) signatureTokenHashes(tokenHashes []uint64, start, length int) []uint64 {
	if length < d.shingleSize || start < 0 || start+length > len(tokenHashes) {
		return nil
	}
	result := make([]uint64, len(d.seeds))
	for i := range result {
		result[i] = math.MaxUint64
	}
	for i := start; i <= start+length-d.shingleSize; i++ {
		shingleHash := uint64(0xcbf29ce484222325)
		for j := i; j < i+d.shingleSize; j++ {
			shingleHash ^= tokenHashes[j]
			shingleHash *= 0x100000001b3
		}
		for j, seed := range d.seeds {
			value := mixJailbreakHash(shingleHash, seed)
			if value < result[j] {
				result[j] = value
			}
		}
	}
	return result
}

func hashJailbreakToken(token string) uint64 {
	hash := uint64(0xcbf29ce484222325)
	for i := 0; i < len(token); i++ {
		hash ^= uint64(token[i])
		hash *= 0x100000001b3
	}
	return hash
}

func mixJailbreakHash(value, seed uint64) uint64 {
	value ^= seed + 0x9e3779b97f4a7c15 + (value << 6) + (value >> 2)
	value ^= value >> 30
	value *= 0xbf58476d1ce4e5b9
	value ^= value >> 27
	value *= 0x94d049bb133111eb
	return value ^ (value >> 31)
}

// LinearJailbreakDetector is a portable CPU L1 trained on safe/unsafe labels.
// It only exits at calibrated high-confidence thresholds; all other inputs use L2.
type LinearJailbreakDetector struct {
	classifier *lineartoken.Classifier
}

func NewLinearJailbreakDetector() (*LinearJailbreakDetector, error) {
	classifier, err := lineartoken.New(jailbreaklinearsvm.Model())
	if err != nil {
		return nil, fmt.Errorf("initialize embedded jailbreak linear model: %w", err)
	}
	return &LinearJailbreakDetector{classifier: classifier}, nil
}

func (d *LinearJailbreakDetector) Predict(text string) (decision string, confidence float32) {
	_, positive, probability := d.classifier.Predict(text)
	confidence = float32(probability)
	if positive {
		return "unsafe", confidence
	}
	return "benign", 1 - confidence
}

func (d *LinearJailbreakDetector) ClassifyWithConfidence(text string, threshold float64) (decision string, confidence float32) {
	decision, confidence = d.Predict(text)
	if threshold <= 0 || threshold >= 1 {
		threshold = 0.9
	}
	if float64(confidence) < threshold {
		return "", confidence
	}
	return decision, confidence
}
