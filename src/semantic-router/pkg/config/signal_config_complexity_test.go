package config

import "testing"

func TestComplexityRuleEffectiveMatchModeDefaultsToEmb(t *testing.T) {
	rule := ComplexityRule{}
	if got := rule.EffectiveMatchMode(); got != ComplexityMatchModeEmb {
		t.Fatalf("expected default complexity mode %q, got %q", ComplexityMatchModeEmb, got)
	}
}

func TestComplexityRuleEffectiveMatchModeKeywordFallbackEmb(t *testing.T) {
	rule := ComplexityRule{MatchMode: " KEYWORD_FALLBACK_EMB "}
	if got := rule.EffectiveMatchMode(); got != ComplexityMatchModeKeywordFallbackEmb {
		t.Fatalf("expected normalized complexity mode %q, got %q", ComplexityMatchModeKeywordFallbackEmb, got)
	}
	if !rule.UseKeywordFallbackToEmb() {
		t.Fatal("expected keyword fallback emb helper to be enabled")
	}
}

func TestComplexityRuleEffectiveMatchModeFallsBackOnUnknownMode(t *testing.T) {
	rule := ComplexityRule{MatchMode: "unknown"}
	if got := rule.EffectiveMatchMode(); got != ComplexityMatchModeEmb {
		t.Fatalf("expected unknown complexity mode to fallback to %q, got %q", ComplexityMatchModeEmb, got)
	}
}
