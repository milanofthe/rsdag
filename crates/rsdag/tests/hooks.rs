//! A consumer's logger and clock, installed once; rsdag reports its compile
//! stages through them and nothing else.

use std::sync::Mutex;

use rsdag::hooks::{self, Clock, Level, Log};
use rsdag::{Graph, Tape, F64};

struct Recorder(Mutex<Vec<(Level, String)>>);
impl Log for Recorder {
    fn log(&self, level: Level, message: &str) {
        self.0.lock().unwrap().push((level, message.to_string()));
    }
}

struct Frozen;
impl Clock for Frozen {
    fn now_ns(&self) -> u64 {
        42
    }
}

static SINK: Recorder = Recorder(Mutex::new(Vec::new()));
static CLOCK: Frozen = Frozen;

#[test]
fn a_compile_reports_its_stages_through_the_installed_hooks() {
    hooks::set_log(&SINK);
    hooks::set_clock(&CLOCK);

    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let e = g.sin(x);
    let s = match g.node(x) {
        rsdag::Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let _ = Tape::compile(&g, &[e], &[s]);

    let got = SINK.0.lock().unwrap().clone();
    let stages: Vec<&str> = got.iter().map(|(_, m)| m.as_str()).collect();
    for want in ["tape analyze", "tape lower", "tape schedule", "tape emit"] {
        assert!(
            stages.iter().any(|m| m.starts_with(want)),
            "no report for {want}: {stages:?}"
        );
    }
    // The frozen clock makes every stage take zero, and every report is
    // debug level: nothing a consumer would see without asking.
    assert!(
        got.iter()
            .all(|(l, m)| *l == Level::Debug && m.ends_with(": 0 ns")),
        "{got:?}"
    );
}
