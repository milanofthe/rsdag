//! The dynamic topological order of the accumulator fusion (see
//! `Program::fuse_accumulators`): whether a kernel can read a value
//! without a cycle, kept answerable as kernels take accumulators.

use std::collections::BTreeSet;

use rustc_hash::FxHashMap as HashMap;

use super::compile::{Kind, Program, Ref};
use crate::constants::ORDER_RELABEL_GROWTH;

const STAMPS_PER_SEARCH: u32 = 2;

// Two-way search incremental topological order: https://doi.org/10.1145/2071379.2071382
pub(super) struct Topo {
    order: Order,
    reader_start: Vec<u32>,
    readers: Vec<u32>,
    readers_added: HashMap<u32, Vec<u32>>,
    seen: Vec<u32>,
    stamp: u32,
    forward_stack: Vec<(u32, u32)>,
    backward_stack: Vec<(u32, u32)>,
    fwd: Vec<u32>,
    back: Vec<u32>,
}

impl Topo {
    // https://github.com/milanofthe/rsdag/issues/241
    pub(super) fn new(
        p: &Program,
        hints: &[(u32, u32)],
        k: u32,
        taken_before_order: &[u32],
    ) -> Topo {
        debug_assert!(hints.is_sorted());
        let m = p.insts.len();
        let for_each_distinct_read_edge = |f: &mut dyn FnMut(u32, u32)| {
            let mut last = vec![u32::MAX; m];
            for i in 0..m {
                for r in p.ins(i) {
                    if let Ref::Value(j, _) = *r {
                        if last[j as usize] != i as u32 {
                            last[j as usize] = i as u32;
                            f(j, i as u32);
                        }
                    }
                }
            }
            for &a in taken_before_order {
                if last[a as usize] != k {
                    last[a as usize] = k;
                    f(a, k);
                }
            }
        };
        let mut count = vec![0u32; m + 1];
        for_each_distinct_read_edge(&mut |j, _| count[j as usize + 1] += 1);
        for i in 0..m {
            count[i + 1] += count[i];
        }
        let mut fill = count.clone();
        let mut readers = vec![0u32; count[m] as usize];
        for_each_distinct_read_edge(&mut |j, i| {
            readers[fill[j as usize] as usize] = i;
            fill[j as usize] += 1;
        });
        let kernel = |i: usize| matches!(p.insts[i].kind, Kind::Gemv { .. } | Kind::Gemm { .. });
        let mut pending = vec![0u32; m];
        for &r in &readers {
            pending[r as usize] += 1;
        }
        let mut hinted = vec![0u32; m];
        for &(_, k) in hints {
            hinted[k as usize] += 1;
        }
        let hints_of = |x: u32| {
            let lo = hints.partition_point(|&(y, _)| y < x);
            let hi = hints.partition_point(|&(y, _)| y <= x);
            &hints[lo..hi]
        };
        let mut ready: Vec<u32> = Vec::new();
        let mut kernels_hints_closed: BTreeSet<u32> = BTreeSet::new();
        let mut kernels_by_open_hints: BTreeSet<(u32, u32)> = BTreeSet::new();
        let release = |r: u32,
                       hinted: &[u32],
                       ready: &mut Vec<u32>,
                       kernels_hints_closed: &mut BTreeSet<u32>,
                       kernels_by_open_hints: &mut BTreeSet<(u32, u32)>| {
            if !kernel(r as usize) {
                ready.push(r);
            } else if hinted[r as usize] == 0 {
                kernels_hints_closed.insert(r);
            } else {
                kernels_by_open_hints.insert((hinted[r as usize], r));
            }
        };
        for i in (0..m as u32).rev() {
            if pending[i as usize] == 0 {
                release(
                    i,
                    &hinted,
                    &mut ready,
                    &mut kernels_hints_closed,
                    &mut kernels_by_open_hints,
                );
            }
        }
        let mut at: Vec<u32> = Vec::with_capacity(m);
        while let Some(i) = ready
            .pop()
            .or_else(|| kernels_hints_closed.pop_first())
            .or_else(|| kernels_by_open_hints.pop_first().map(|(_, k)| k))
        {
            at.push(i);
            for &(_, k) in hints_of(i) {
                let h = hinted[k as usize];
                hinted[k as usize] = h - 1;
                if kernels_by_open_hints.remove(&(h, k)) {
                    if h == 1 {
                        kernels_hints_closed.insert(k);
                    } else {
                        kernels_by_open_hints.insert((h - 1, k));
                    }
                }
            }
            for &r in &readers[count[i as usize] as usize..count[i as usize + 1] as usize] {
                pending[r as usize] -= 1;
                if pending[r as usize] == 0 {
                    release(
                        r,
                        &hinted,
                        &mut ready,
                        &mut kernels_hints_closed,
                        &mut kernels_by_open_hints,
                    );
                }
            }
        }
        debug_assert_eq!(at.len(), m, "the lowered program is acyclic");
        Topo {
            order: Order::new(&at),
            reader_start: count,
            readers,
            readers_added: HashMap::default(),
            seen: vec![0; m],
            stamp: 0,
            forward_stack: Vec::new(),
            backward_stack: Vec::new(),
            fwd: Vec::new(),
            back: Vec::new(),
        }
    }

    // https://github.com/milanofthe/rsdag/issues/241
    pub(super) fn read(&mut self, p: &Program, k: u32, a: u32) -> bool {
        let n = self.seen.len() as u32;
        let a_is_constant_added_after_order = a >= n;
        if a_is_constant_added_after_order {
            return true;
        }
        let label = &self.order.label;
        let (lb, ub) = (label[k as usize], label[a as usize]);
        if ub < lb {
            self.readers_added.entry(a).or_default().push(k);
            return true;
        }
        if self.stamp > u32::MAX - 2 * STAMPS_PER_SEARCH {
            self.seen.fill(0);
            self.stamp = 0;
        }
        self.stamp += STAMPS_PER_SEARCH;
        let (forward_mark, backward_mark) = (self.stamp, self.stamp + 1);
        self.forward_stack.clear();
        self.backward_stack.clear();
        self.fwd.clear();
        self.back.clear();
        self.forward_stack.push((k, 0));
        self.fwd.push(k);
        self.seen[k as usize] = forward_mark;
        self.backward_stack.push((a, 0));
        self.back.push(a);
        self.seen[a as usize] = backward_mark;
        let backward = loop {
            let Some((w, e)) = self.forward_stack.pop() else {
                break false;
            };
            let (s0, s1) = (
                self.reader_start[w as usize],
                self.reader_start[w as usize + 1],
            );
            let deg = s1 - s0;
            let r = if e < deg {
                Some(self.readers[(s0 + e) as usize])
            } else {
                self.readers_added
                    .get(&w)
                    .and_then(|v| v.get((e - deg) as usize))
                    .copied()
            };
            if let Some(r) = r {
                self.forward_stack.push((w, e + 1));
                if r == a {
                    return false;
                }
                if self.seen[r as usize] != forward_mark && label[r as usize] < ub {
                    self.seen[r as usize] = forward_mark;
                    self.fwd.push(r);
                    self.forward_stack.push((r, 0));
                }
            }
            let Some((w, e)) = self.backward_stack.pop() else {
                break true;
            };
            if let Some(&r) = p.ins(w as usize).get(e as usize) {
                self.backward_stack.push((w, e + 1));
                if let Ref::Value(j, _) = r {
                    if j == k {
                        return false;
                    }
                    if j < n && self.seen[j as usize] != backward_mark && label[j as usize] > lb {
                        self.seen[j as usize] = backward_mark;
                        self.back.push(j);
                        self.backward_stack.push((j, 0));
                    }
                }
            }
        };
        let order = &mut self.order;
        let set = if backward {
            &mut self.back
        } else {
            &mut self.fwd
        };
        set.sort_unstable_by_key(|&w| order.label[w as usize]);
        for &w in set.iter() {
            order.unlink(w);
        }
        let mut at = if backward { order.prev[k as usize] } else { a };
        for &w in set.iter() {
            order.insert_after(at, w);
            at = w;
        }
        self.readers_added.entry(a).or_default().push(k);
        true
    }
}

// Order maintenance by labels: https://doi.org/10.1007/3-540-45749-6_17
struct Order {
    label: Vec<u64>,
    prev: Vec<u32>,
    next: Vec<u32>,
}

impl Order {
    fn new(order: &[u32]) -> Order {
        let n = order.len();
        let (head, tail) = (n as u32, n as u32 + 1);
        let mut label = vec![0u64; n + 2];
        let mut prev = vec![0u32; n + 2];
        let mut next = vec![0u32; n + 2];
        let step = u64::MAX / (n as u64 + 2);
        let mut last = head;
        for (q, &w) in order.iter().enumerate() {
            label[w as usize] = (q as u64 + 1) * step;
            prev[w as usize] = last;
            next[last as usize] = w;
            last = w;
        }
        next[last as usize] = tail;
        prev[tail as usize] = last;
        label[tail as usize] = u64::MAX;
        Order { label, prev, next }
    }

    fn unlink(&mut self, w: u32) {
        let (p, q) = (self.prev[w as usize], self.next[w as usize]);
        self.next[p as usize] = q;
        self.prev[q as usize] = p;
    }

    fn insert_after(&mut self, x: u32, w: u32) {
        let y = self.next[x as usize];
        self.prev[w as usize] = x;
        self.next[w as usize] = y;
        self.next[x as usize] = w;
        self.prev[y as usize] = w;
        let (lo, hi) = (self.label[x as usize], self.label[y as usize]);
        if hi - lo >= 2 {
            self.label[w as usize] = lo + (hi - lo) / 2;
            return;
        }
        let head = self.label.len() as u32 - 2;
        let tail = head + 1;
        for i in 1..=u64::BITS {
            let size = 1u128 << i;
            let base = lo as u128 & !(size - 1);
            let end = base + size;
            let mut first = w;
            let mut c = 1u128;
            let mut v = x;
            while v != head && self.label[v as usize] as u128 >= base {
                first = v;
                c += 1;
                v = self.prev[v as usize];
            }
            let mut v = y;
            while v != tail && (self.label[v as usize] as u128) < end {
                c += 1;
                v = self.next[v as usize];
            }
            let range_sparse_enough = (c as f64) < ORDER_RELABEL_GROWTH.powi(i as i32);
            if i == u64::BITS || range_sparse_enough {
                let from = base.max(1);
                let to = end.min(u64::MAX as u128);
                let gap = (to - from) / (c + 1);
                let mut v = first;
                for j in 0..c {
                    self.label[v as usize] = (from + gap * (j + 1)) as u64;
                    v = self.next[v as usize];
                }
                return;
            }
        }
    }
}
