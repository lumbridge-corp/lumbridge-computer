//! A small parser for the Prometheus text exposition format (0.0.4).
//!
//! Engines are the authority for their own load, and they all speak this
//! format. We need a parser rather than substring matching because label
//! values carry arbitrary text: vLLM's `cache_config_info` holds lists like
//! `"[]"`, and a caller label may hold quotes or commas. A line we cannot
//! parse is skipped instead of failing the whole scrape: one odd series from a
//! newer engine build must not blind telemetry to everything else.

use std::collections::BTreeMap;

/// One sample line: metric name, its labels and value. The optional
/// exposition timestamp is validated and dropped; every engine we scrape
/// leaves it out, and the scrape time is what we record.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

pub fn parse(text: &str) -> Vec<Sample> {
    text.lines().filter_map(parse_line).collect()
}

/// `None` for comments, blank lines and anything malformed.
pub fn parse_line(line: &str) -> Option<Sample> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let name_end = line.find(|c: char| c == '{' || c.is_whitespace())?;
    let name = &line[..name_end];
    if !valid_name(name, true) {
        return None;
    }
    let mut rest = &line[name_end..];
    let labels = match rest.strip_prefix('{') {
        Some(inner) => {
            let (labels, after) = parse_labels(inner)?;
            rest = after;
            labels
        }
        None => BTreeMap::new(),
    };
    let mut fields = rest.split_whitespace();
    let value = parse_value(fields.next()?)?;
    if let Some(timestamp) = fields.next() {
        timestamp.parse::<i64>().ok()?;
    }
    if fields.next().is_some() {
        return None;
    }
    Some(Sample {
        name: name.to_string(),
        labels,
        value,
    })
}

/// Go's `strconv.ParseFloat` spellings, which is what the format specifies:
/// `+Inf`, `-Inf`, `NaN` and exponent notation such as `1.79e+09`.
pub fn parse_value(text: &str) -> Option<f64> {
    let unsigned = text
        .strip_prefix('+')
        .or_else(|| text.strip_prefix('-'))
        .unwrap_or(text);
    if unsigned.eq_ignore_ascii_case("inf") || unsigned.eq_ignore_ascii_case("infinity") {
        return Some(if text.starts_with('-') {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if text.eq_ignore_ascii_case("nan") {
        return Some(f64::NAN);
    }
    // Rust also accepts "inf"/"nan" spellings we already handled; anything
    // else it accepts is a plain decimal or exponent float.
    text.parse::<f64>().ok()
}

/// Parse `key="value",...}` and return the labels plus the text after `}`.
fn parse_labels(input: &str) -> Option<(BTreeMap<String, String>, &str)> {
    let mut labels = BTreeMap::new();
    let mut chars = input.char_indices().peekable();
    loop {
        // Separators: whitespace, and the comma (a trailing one is legal).
        while chars
            .peek()
            .is_some_and(|(_, c)| c.is_whitespace() || *c == ',')
        {
            chars.next();
        }
        let (start, first) = chars.next()?;
        match first {
            '}' => return Some((labels, &input[start + 1..])),
            '=' => return None,
            _ => {}
        }
        let mut key_end = start + first.len_utf8();
        loop {
            let (index, c) = chars.next()?;
            if c == '=' {
                break;
            }
            key_end = index + c.len_utf8();
        }
        let key = input[start..key_end].trim();
        if !valid_name(key, false) {
            return None;
        }
        while chars.peek().is_some_and(|(_, c)| c.is_whitespace()) {
            chars.next();
        }
        if chars.next()?.1 != '"' {
            return None;
        }
        let mut value = String::new();
        loop {
            match chars.next()?.1 {
                '"' => break,
                '\\' => match chars.next()?.1 {
                    'n' => value.push('\n'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    // Not a defined escape: keep it literally rather than
                    // guessing what the exporter meant.
                    other => {
                        value.push('\\');
                        value.push(other);
                    }
                },
                c => value.push(c),
            }
        }
        labels.insert(key.to_string(), value);
        while chars.peek().is_some_and(|(_, c)| c.is_whitespace()) {
            chars.next();
        }
        match chars.peek() {
            Some((_, ',')) | Some((_, '}')) => {}
            _ => return None,
        }
    }
}

/// Metric names may contain `:`; label names may not.
fn valid_name(name: &str, metric: bool) -> bool {
    let mut chars = name.chars();
    let ok_first = |c: char| c.is_ascii_alphabetic() || c == '_' || (metric && c == ':');
    let ok_rest = |c: char| c.is_ascii_alphanumeric() || c == '_' || (metric && c == ':');
    chars.next().is_some_and(ok_first) && chars.all(ok_rest)
}

/// Sum the finite values of every series named `name` that `keep` accepts.
/// `None` when there is no such sample at all, so callers can tell "absent"
/// (unknown) from "zero" (idle). NaN/Inf samples count as absent.
pub fn sum_where<F>(samples: &[Sample], name: &str, keep: F) -> Option<f64>
where
    F: Fn(&Sample) -> bool,
{
    samples
        .iter()
        .filter(|sample| sample.name == name && sample.value.is_finite() && keep(sample))
        .map(|sample| sample.value)
        .reduce(|sum, value| sum + value)
}

pub fn sum(samples: &[Sample], name: &str) -> Option<f64> {
    sum_where(samples, name, |_| true)
}

/// Largest finite value of `name`, for ratios that must not be summed.
pub fn max_where<F>(samples: &[Sample], name: &str, keep: F) -> Option<f64>
where
    F: Fn(&Sample) -> bool,
{
    samples
        .iter()
        .filter(|sample| sample.name == name && sample.value.is_finite() && keep(sample))
        .map(|sample| sample.value)
        .reduce(f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_blank_lines_and_help_text_are_skipped() {
        let samples = parse("# HELP x A help line with {braces} 1\n# TYPE x gauge\n\n  \nx 1\n");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].name, "x");
        assert!(samples[0].labels.is_empty());
    }

    #[test]
    fn label_values_keep_escaped_quotes_commas_braces_and_spaces() {
        let sample = parse_line(
            r#"sglang:num_requests_total{agent="prime\"one",note="a, b} c",path="C:\\x",nl="1\n2",} 12"#,
        )
        .unwrap();
        assert_eq!(sample.name, "sglang:num_requests_total");
        assert_eq!(sample.labels["agent"], "prime\"one");
        assert_eq!(sample.labels["note"], "a, b} c");
        assert_eq!(sample.labels["path"], "C:\\x");
        assert_eq!(sample.labels["nl"], "1\n2");
        assert_eq!(sample.value, 12.0);
    }

    #[test]
    fn non_ascii_label_values_survive_intact() {
        let sample = parse_line("m{who=\"Kärti ✓\"} 1").unwrap();
        assert_eq!(sample.labels["who"], "Kärti ✓");
    }

    #[test]
    fn special_floats_and_exponents_parse() {
        assert_eq!(parse_value("1.79e+09"), Some(1.79e9));
        assert_eq!(parse_value("3.0991928e+07"), Some(30_991_928.0));
        assert_eq!(parse_value("-2E-3"), Some(-0.002));
        assert_eq!(parse_value("+Inf"), Some(f64::INFINITY));
        assert_eq!(parse_value("-Inf"), Some(f64::NEG_INFINITY));
        assert!(parse_value("NaN").unwrap().is_nan());
        assert_eq!(parse_value("one"), None);
        let bucket = parse_line("h_bucket{le=\"+Inf\"} 7").unwrap();
        assert_eq!(bucket.labels["le"], "+Inf");
        assert_eq!(bucket.value, 7.0);
    }

    #[test]
    fn an_exposition_timestamp_is_not_mistaken_for_the_value() {
        let sample = parse_line("vllm:num_requests_running{engine=\"0\"} 2 1793740525000").unwrap();
        assert_eq!(sample.value, 2.0);
        assert_eq!(parse_line("x 2 notatime"), None);
        assert_eq!(parse_line("x 2 3 4"), None);
    }

    #[test]
    fn malformed_lines_are_skipped_not_fatal() {
        let samples = parse("x{a=\"1\" 2\ny{a=1} 3\n9bad 1\nnovalue\nz{a=\"1\"} 4\n");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].name, "z");
    }

    #[test]
    fn sums_ignore_nan_and_tell_absent_from_zero() {
        let samples = parse("g{e=\"0\"} 1\ng{e=\"1\"} 2\ng{e=\"2\"} NaN\nz 0\nn NaN\n");
        assert_eq!(sum(&samples, "g"), Some(3.0));
        assert_eq!(sum(&samples, "z"), Some(0.0));
        assert_eq!(sum(&samples, "missing"), None);
        assert_eq!(sum(&samples, "n"), None);
        assert_eq!(max_where(&samples, "g", |_| true), Some(2.0));
    }
}
