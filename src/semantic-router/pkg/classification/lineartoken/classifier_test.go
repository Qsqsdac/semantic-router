package lineartoken

import "testing"

func TestClassifierUsesBinaryTokenFeatures(t *testing.T) {
	classifier, err := New(Model{
		Tokens:        []string{"ignore", "policy"},
		Weights:       []float64{0.7, 0.6},
		Intercept:     -1,
		NegativeLabel: "safe",
		PositiveLabel: "unsafe",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	label, positive, probability := classifier.Predict("Ignore the policy; ignore it again.")
	if label != "unsafe" || !positive || probability <= 0.5 {
		t.Fatalf("Predict() = (%q, %v, %v), want unsafe positive probability", label, positive, probability)
	}
	if score := classifier.Score("ignored policy"); score != -0.4 {
		t.Fatalf("Score() = %v, want -0.4; token boundaries should prevent partial matching", score)
	}
}
