package classification

import "testing"

func TestSplitComplexitySentences(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		expected []string
	}{
		{
			name:     "empty",
			query:    "   ",
			expected: nil,
		},
		{
			name:     "single sentence",
			query:    "summarize this paragraph",
			expected: []string{"summarize this paragraph"},
		},
		{
			name:  "mixed punctuation",
			query: "First sentence, still first: part A. Second sentence! 第三句？\nFourth; fifth；, sixth，: seventh：",
			expected: []string{
				"First sentence",
				"still first",
				"part A",
				"Second sentence",
				"第三句",
				"Fourth",
				"fifth",
				"sixth",
				"seventh",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := splitComplexitySentences(tt.query)
			if len(actual) != len(tt.expected) {
				t.Fatalf("expected len=%d, got=%d (%v)", len(tt.expected), len(actual), actual)
			}
			for i := range actual {
				if actual[i] != tt.expected[i] {
					t.Fatalf("expected[%d]=%q, got=%q", i, tt.expected[i], actual[i])
				}
			}
		})
	}
}
