package config

import "testing"

func TestPreferenceModelConfigWithDefaultsEnablesContrastiveByDefault(t *testing.T) {
	cfg := PreferenceModelConfig{}.WithDefaults()
	if cfg.UseContrastive == nil || !*cfg.UseContrastive {
		t.Fatal("expected default preference config to enable contrastive mode")
	}
}

func TestPreferenceModelConfigWithDefaultsPreservesExplicitFalse(t *testing.T) {
	disabled := false
	cfg := PreferenceModelConfig{UseContrastive: &disabled}.WithDefaults()
	if cfg.UseContrastive == nil {
		t.Fatal("expected explicit false preference config to be preserved")
	}
	if *cfg.UseContrastive {
		t.Fatal("expected explicit false preference config to remain disabled")
	}
}

func TestCategoryModelEffectiveIntentMatchModeDefaultsToBERT(t *testing.T) {
	cfg := CategoryModel{}
	if got := cfg.EffectiveIntentMatchMode(); got != IntentMatchModeBERT {
		t.Fatalf("expected default intent mode %q, got %q", IntentMatchModeBERT, got)
	}
}

func TestCategoryModelEffectiveIntentMatchModeNormalizesFallbackMode(t *testing.T) {
	cfg := CategoryModel{IntentMatchMode: " KEYWORD_FALLBACK_BERT "}
	if got := cfg.EffectiveIntentMatchMode(); got != IntentMatchModeKeywordFallbackBERT {
		t.Fatalf("expected normalized intent mode %q, got %q", IntentMatchModeKeywordFallbackBERT, got)
	}
	if !cfg.UseKeywordFallbackToBERT() {
		t.Fatal("expected keyword fallback mode to be enabled")
	}
}

func TestCategoryModelEffectiveIntentMatchModeNormalizesFastTextMode(t *testing.T) {
	cfg := CategoryModel{IntentMatchMode: " FASTTEXT_FALLBACK_BERT "}
	if got := cfg.EffectiveIntentMatchMode(); got != IntentMatchModeFastTextFallbackBERT {
		t.Fatalf("expected normalized intent mode %q, got %q", IntentMatchModeFastTextFallbackBERT, got)
	}
	if !cfg.UseFastTextFallbackToBERT() {
		t.Fatal("expected fastText fallback mode to be enabled")
	}
}

func TestCategoryModelEffectiveIntentMatchModeFallsBackOnUnknownMode(t *testing.T) {
	cfg := CategoryModel{IntentMatchMode: "unknown"}
	if got := cfg.EffectiveIntentMatchMode(); got != IntentMatchModeBERT {
		t.Fatalf("expected unknown mode to fallback to %q, got %q", IntentMatchModeBERT, got)
	}
}
