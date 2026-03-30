use std::collections::{HashMap, HashSet};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

#[derive(Debug, Clone)]
struct RuleEntry {
    name: String,
    total_keywords: usize,
}

#[derive(Debug, Clone)]
struct PatternMeta {
    rule_idx: usize,
    keyword: String,
}

#[derive(Debug, Clone)]
pub struct AhoClassification {
    pub rule_name: String,
    pub matched_keywords: Vec<String>,
    pub scores: Vec<f32>,
    pub match_count: usize,
    pub total_keywords: usize,
}

#[derive(Default)]
pub struct AhoClassifier {
    rules: Vec<RuleEntry>,
    sensitive_patterns: Vec<String>,
    insensitive_patterns: Vec<String>,
    sensitive_meta: Vec<PatternMeta>,
    insensitive_meta: Vec<PatternMeta>,
    sensitive_matcher: Option<AhoCorasick>,
    insensitive_matcher: Option<AhoCorasick>,
}

impl AhoClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_rule(&mut self, name: String, keywords: Vec<String>, case_sensitive: bool) {
        let mut unique_keywords = HashSet::new();
        for kw in keywords {
            let trimmed = kw.trim();
            if trimmed.is_empty() {
                continue;
            }
            unique_keywords.insert(trimmed.to_string());
        }

        if unique_keywords.is_empty() {
            return;
        }

        let rule_idx = self.rules.len();
        self.rules.push(RuleEntry {
            name,
            total_keywords: unique_keywords.len(),
        });

        for keyword in unique_keywords {
            if case_sensitive {
                self.sensitive_patterns.push(keyword.clone());
                self.sensitive_meta.push(PatternMeta { rule_idx, keyword });
            } else {
                self.insensitive_patterns.push(keyword.to_lowercase());
                self.insensitive_meta
                    .push(PatternMeta { rule_idx, keyword });
            }
        }

        self.rebuild_matchers();
    }

    fn rebuild_matchers(&mut self) {
        self.sensitive_matcher = build_matcher(&self.sensitive_patterns);
        self.insensitive_matcher = build_matcher(&self.insensitive_patterns);
    }

    pub fn classify(&self, text: &str) -> Option<AhoClassification> {
        if text.is_empty() {
            return None;
        }

        let mut matches_by_rule: HashMap<usize, HashSet<String>> = HashMap::new();

        if let Some(matcher) = &self.sensitive_matcher {
            for found in matcher.find_iter(text) {
                if !is_whole_word_match(text, found.start(), found.end()) {
                    continue;
                }
                let idx = found.pattern().as_usize();
                if let Some(meta) = self.sensitive_meta.get(idx) {
                    matches_by_rule
                        .entry(meta.rule_idx)
                        .or_default()
                        .insert(meta.keyword.clone());
                }
            }
        }

        if let Some(matcher) = &self.insensitive_matcher {
            let lower = text.to_lowercase();
            for found in matcher.find_iter(&lower) {
                if !is_whole_word_match(&lower, found.start(), found.end()) {
                    continue;
                }
                let idx = found.pattern().as_usize();
                if let Some(meta) = self.insensitive_meta.get(idx) {
                    matches_by_rule
                        .entry(meta.rule_idx)
                        .or_default()
                        .insert(meta.keyword.clone());
                }
            }
        }

        let mut best: Option<(usize, usize, f32)> = None;
        for (rule_idx, matched) in &matches_by_rule {
            let match_count = matched.len();
            if match_count == 0 {
                continue;
            }

            let total = self
                .rules
                .get(*rule_idx)
                .map(|r| r.total_keywords)
                .unwrap_or(1)
                .max(1);
            let score = match_count as f32 / total as f32;

            match best {
                Some((best_idx, best_count, best_score)) => {
                    if match_count > best_count
                        || (match_count == best_count && score > best_score)
                        || (match_count == best_count
                            && (score - best_score).abs() < f32::EPSILON
                            && *rule_idx < best_idx)
                    {
                        best = Some((*rule_idx, match_count, score));
                    }
                }
                None => best = Some((*rule_idx, match_count, score)),
            }
        }

        let (rule_idx, match_count, rule_score) = best?;
        let rule = self.rules.get(rule_idx)?;
        let mut matched_keywords: Vec<String> = matches_by_rule
            .remove(&rule_idx)
            .unwrap_or_default()
            .into_iter()
            .collect();
        matched_keywords.sort();

        Some(AhoClassification {
            rule_name: rule.name.clone(),
            scores: vec![rule_score; matched_keywords.len()],
            matched_keywords,
            match_count,
            total_keywords: rule.total_keywords,
        })
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_whole_word_match(text: &str, start: usize, end: usize) -> bool {
    let left_ok = if start == 0 {
        true
    } else {
        text[..start]
            .chars()
            .next_back()
            .map_or(true, |c| !is_word_char(c))
    };

    let right_ok = if end >= text.len() {
        true
    } else {
        text[end..]
            .chars()
            .next()
            .map_or(true, |c| !is_word_char(c))
    };

    left_ok && right_ok
}

fn build_matcher(patterns: &[String]) -> Option<AhoCorasick> {
    if patterns.is_empty() {
        return None;
    }

    AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns)
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_word_prevents_substring_false_positive() {
        let mut classifier = AhoClassifier::new();
        classifier.add_rule("other".to_string(), vec!["Art".to_string()], false);
        classifier.add_rule("math".to_string(), vec!["Ratio".to_string()], false);

        let text = "particularly due to corporate policy";
        let result = classifier.classify(text);
        assert!(result.is_none(), "substring hits should not be counted");
    }

    #[test]
    fn whole_word_keeps_valid_keyword_hits() {
        let mut classifier = AhoClassifier::new();
        classifier.add_rule("engineering".to_string(), vec!["CAD".to_string()], false);

        let result = classifier.classify("what is cad?");
        assert!(result.is_some(), "whole-word keyword should match");
        let matched = result.expect("expected match result");
        assert_eq!(matched.rule_name, "engineering");
    }

    #[test]
    fn whole_word_supports_multi_word_phrase() {
        let mut classifier = AhoClassifier::new();
        classifier.add_rule(
            "business".to_string(),
            vec!["Supply chain".to_string()],
            false,
        );

        let result = classifier.classify("how to optimize supply chain efficiency");
        assert!(
            result.is_some(),
            "multi-word phrase should match as whole phrase"
        );
        let matched = result.expect("expected phrase match result");
        assert_eq!(matched.rule_name, "business");
    }
}
