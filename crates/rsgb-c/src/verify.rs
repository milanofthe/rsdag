//! Compile-and-compare verification: build the emitted C with the host
//! compiler, run it on given inputs, and return the outputs so a test can
//! compare them with the interpreter.

use std::path::PathBuf;
use std::process::Command;

use rsgb::Tape;

/// The host C compiler, if one is on the path (`cc`, then `clang`, `gcc`).
pub fn find_compiler() -> Option<String> {
    for cc in ["cc", "clang", "gcc"] {
        if Command::new(cc)
            .arg("--version")
            .output()
            .map_or(false, |o| o.status.success())
        {
            return Some(cc.to_string());
        }
    }
    None
}

/// Emit `tape`, compile it with `cc` together with a harness that evaluates
/// every row of `inputs`, run it, and parse the outputs (transported as hex
/// bit patterns, so nothing is lost in printing).
pub fn run_c(cc: &str, tape: &Tape, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    run_c_with(cc, tape, inputs, false).map(|(rows, _)| rows)
}

/// [`run_c`] that also returns the harness's raw stdout; with `trace_slots`
/// the harness prints every work slot after the last row (debugging a
/// mismatch op by op).
pub fn run_c_with(
    cc: &str,
    tape: &Tape,
    inputs: &[Vec<f64>],
    trace_slots: bool,
) -> Result<(Vec<Vec<f64>>, String), String> {
    let src = crate::emit(tape, "rsgb_fn").map_err(|e| e.to_string())?;
    let n_in = inputs.first().map_or(0, |r| r.len());
    let n_out = tape.n_outputs();
    let mut main = String::new();
    main.push_str("#include <stdio.h>\n");
    main.push_str(&src);
    main.push_str("int main(void) {\n");
    main.push_str(&format!(
        "    static double work[{}];\n    double out[{}];\n",
        tape.n_slots().max(1),
        n_out.max(1)
    ));
    for row in inputs {
        main.push_str(&format!("    {{ const double in[{}] = {{", n_in.max(1)));
        for (k, v) in row.iter().enumerate() {
            if k > 0 {
                main.push_str(", ");
            }
            main.push_str(&crate::lit_pub(*v));
        }
        if row.is_empty() {
            main.push_str("0.0");
        }
        main.push_str("};\n      rsgb_fn(in, work, out);\n");
        main.push_str(&format!(
            "      for (int k = 0; k < {n_out}; k++) {{ uint64_t b; memcpy(&b, &out[k], 8); printf(\"%016llx \", (unsigned long long)b); }}\n      printf(\"\\n\"); }}\n"
        ));
        if trace_slots {
            main.push_str(&format!(
                "    for (int k = 0; k < {}; k++) {{ uint64_t b; memcpy(&b, &work[k], 8); printf(\"slot %d %016llx %.17g\\n\", k, (unsigned long long)b, work[k]); }}\n",
                tape.n_slots()
            ));
        }
    }
    main.push_str("    return 0;\n}\n");
    let dir = std::env::temp_dir().join(format!("rsgb-c-{}-{}", std::process::id(), tape.n_ops()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let c_path: PathBuf = dir.join("harness.c");
    let bin: PathBuf = dir.join("harness");
    std::fs::write(&c_path, main).map_err(|e| e.to_string())?;
    let out = Command::new(cc)
        .args(["-O1", "-ffp-contract=off", "-o"])
        .arg(&bin)
        .arg(&c_path)
        .arg("-lm")
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "cc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let run = Command::new(&bin).output().map_err(|e| e.to_string())?;
    if !run.status.success() {
        return Err("harness failed".to_string());
    }
    let text = String::from_utf8_lossy(&run.stdout).to_string();
    let rows = text
        .lines()
        .filter(|l| !l.starts_with("slot "))
        .map(|l| {
            l.split_whitespace()
                .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap_or(0)))
                .collect()
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    Ok((rows, text))
}
