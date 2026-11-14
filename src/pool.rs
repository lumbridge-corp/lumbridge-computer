//! Memory pools: where a model's bytes live, and the numbers admission guards.
//!
//! A [`Pool`] is a named piece of memory with a budget, a safety margin and a
//! watchdog floor. Admission, headroom, reservations and the watchdog all ask a
//! pool, never `/proc/meminfo` directly, so a box with a second pool (a discrete
//! GPU's VRAM next to host RAM) is a new [`PoolId`] variant and its sensing,
//! not a rewrite of the governor.
//!
//! The DGX Spark has exactly one pool: `unified`, the 128 GB that GB10's CPU and
//! GPU share, sensed from `/proc/meminfo` (NVML reports no memory on GB10).
//! Every model and every reservation belongs to it unless the registry says
//! otherwise, and today the registry can say nothing else.

use crate::config::{Model, Registry, Reservation};
use crate::mem::{self, MemInfo};
use serde::{Deserialize, Serialize};

/// The unified pool's ceiling for model memory when a Scene declares none.
pub const DEFAULT_BUDGET_GB: f64 = 100.0;
/// Extra headroom required before admitting a new model.
pub const SAFETY_MARGIN_GB: f64 = 8.0;
/// If memory available to models dips below this, the watchdog sheds a model.
pub const WATCHDOG_FLOOR_GB: f64 = 3.0;

/// The name a model, reservation or `lumbridge-computer.toml` uses for a pool. An enum
/// rather than a string so that an unknown pool is a load error, not a model
/// silently admitted against memory that does not exist.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum PoolId {
    /// GB10's shared CPU+GPU memory.
    #[default]
    Unified,
}

impl PoolId {
    pub fn name(self) -> &'static str {
        match self {
            PoolId::Unified => "unified",
        }
    }

    /// This build's definition of the pool: the governor's compiled numbers.
    pub fn pool(self) -> Pool {
        match self {
            PoolId::Unified => Pool::unified(),
        }
    }
}

impl std::fmt::Display for PoolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One memory pool and the limits admission enforces on it. Totals and
/// availability are sensed ([`Pool::sense`]), never stored: they change by the second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pool {
    pub id: PoolId,
    /// Ceiling for model memory when a Scene declares no `budget_gb`.
    pub budget_gb: f64,
    pub safety_margin_gb: f64,
    /// The watchdog sheds a model when memory available to models falls below this.
    pub floor_gb: f64,
}

impl Pool {
    pub const fn unified() -> Pool {
        Pool {
            id: PoolId::Unified,
            budget_gb: DEFAULT_BUDGET_GB,
            safety_margin_gb: SAFETY_MARGIN_GB,
            floor_gb: WATCHDOG_FLOOR_GB,
        }
    }

    /// The pool `lumbridge-computer.toml [memory]` describes. Only the box checks use this:
    /// the running governor keeps the compiled numbers, and doctor warns when the
    /// file disagrees with them.
    pub fn from_config(memory: &crate::box_::config::MemorySection) -> Pool {
        Pool {
            id: memory.pool,
            budget_gb: memory.budget_gb,
            safety_margin_gb: memory.safety_margin_gb,
            floor_gb: memory.watchdog_floor_gb,
        }
    }

    /// The same pool with an operator-chosen watchdog floor (`--floor`).
    pub fn with_floor(self, floor_gb: f64) -> Pool {
        Pool { floor_gb, ..self }
    }

    /// Read the pool's total and available memory now.
    pub fn sense(&self) -> anyhow::Result<MemInfo> {
        match self.id {
            // On GB10 the kernel's view of RAM *is* the GPU's memory too.
            PoolId::Unified => mem::read(),
        }
    }

    pub fn holds(&self, model: &Model) -> bool {
        model.pool == self.id
    }

    /// A Scene's declared ceiling, or this pool's default when it declares none.
    pub fn scene_budget_gb(&self, declared: Option<f64>) -> f64 {
        declared.unwrap_or(self.budget_gb)
    }

    fn reservations<'a>(&self, reg: &'a Registry) -> impl Iterator<Item = &'a Reservation> {
        let id = self.id;
        reg.reservations.values().filter(move |r| r.pool == id)
    }

    /// Sum of every reservation's ceiling in this pool.
    pub fn reserved_gb(&self, reg: &Registry) -> f64 {
        self.reservations(reg).map(|r| r.gb).sum()
    }

    /// The part of each reservation its tenant has not claimed yet, summed.
    ///
    /// `MemAvailable` already excludes what a tenant uses today. What it can't see is
    /// the room the tenant was promised to grow into — Postgres under load, a burst of
    /// realtime connections. Subtracting only the *unused* part, rather than the whole
    /// reservation, avoids counting the tenant's current use twice.
    pub fn reserved_unused_gb(&self, reg: &Registry) -> f64 {
        self.reserved_unused_with(reg, |r| r.cgroup.as_deref().and_then(cgroup_used_gb))
    }

    /// `reserved_unused_gb` with the tenant's use supplied, so tests can pin it.
    fn reserved_unused_with(
        &self,
        reg: &Registry,
        used: impl Fn(&Reservation) -> Option<f64>,
    ) -> f64 {
        self.reservations(reg).map(|r| unused_gb(r, used(r))).sum()
    }

    /// Available memory as models see it: minus the room reserved tenants may still grow into.
    pub fn model_available_gb(&self, reg: &Registry, available_gb: f64) -> f64 {
        available_gb - self.reserved_unused_gb(reg)
    }

    /// The declared budget, never above what the pool has once reservations are held.
    /// A Scene's `budget_gb` is typed by a human (the shipped one says 125 GB on a 121.7 GB
    /// Spark); capping it here keeps the declared ceiling meaningful when sensing fails and
    /// makes Scene fit verdicts include the Server.
    pub fn effective_budget_gb(
        &self,
        budget_gb: f64,
        total_gb: Option<f64>,
        reserved_gb: f64,
    ) -> f64 {
        match total_gb {
            Some(total) => budget_gb.min(total - reserved_gb),
            None => budget_gb,
        }
    }

    /// Admission control. Two independent ceilings, and both must hold.
    ///
    /// 1. **Declared** — `committed + add + margin <= budget`.
    /// 2. **Observed** — `add + margin <= available`.
    ///
    /// The second one is what makes the promise real. `budget_gb`, and every
    /// `footprint_gb` feeding `committed_gb`, are numbers a human typed into the
    /// registry. If any of them is optimistic — a model that grows past its declared
    /// footprint, a KV cache larger than expected, anything started outside Lumbridge Computer —
    /// the declared check happily passes while the box is already out of memory, and
    /// on unified memory that ends in a wedged machine rather than a failed malloc.
    /// `available_gb` comes from sensing, which counts reality.
    ///
    /// `available_gb` is `None` when sensing failed. That falls back to the declared
    /// ceiling alone: refusing every start because /proc/meminfo was unreadable would
    /// turn a sensing failure into a total outage.
    pub fn can_admit(
        &self,
        add_gb: f64,
        committed_gb: f64,
        budget_gb: f64,
        available_gb: Option<f64>,
    ) -> bool {
        self.can_admit_observed(add_gb, add_gb, committed_gb, budget_gb, available_gb)
    }

    /// [`Pool::can_admit`] with the two ceilings asked for different sizes.
    ///
    /// The declared ceiling adds up registry numbers, so it takes the model's admission
    /// figure (`add_gb`). The observed ceiling is about the end state of the machine:
    /// after this model is up, at least the safety margin must still be available. What
    /// the model will take out of `available_gb` is its real resident size
    /// (`observed_add_gb`, see `footprint::observed_gb`), not its padded declaration:
    /// sensing already counts every running model at its true size, so padding here
    /// spends the margin twice and makes a set of models that ran fine impossible to
    /// restore after a restart.
    pub fn can_admit_observed(
        &self,
        add_gb: f64,
        observed_add_gb: f64,
        committed_gb: f64,
        budget_gb: f64,
        available_gb: Option<f64>,
    ) -> bool {
        if committed_gb + add_gb + self.safety_margin_gb > budget_gb {
            return false;
        }
        match available_gb {
            Some(available) => observed_add_gb + self.safety_margin_gb <= available,
            None => true,
        }
    }

    /// Headroom under the declared budget alone.
    pub fn declared_headroom_gb(&self, committed_gb: f64, budget_gb: f64) -> f64 {
        budget_gb - committed_gb - self.safety_margin_gb
    }

    /// What admission will actually enforce right now, for reporting. The lower of the
    /// declared headroom and the observed headroom.
    pub fn headroom_gb(&self, committed_gb: f64, budget_gb: f64, available_gb: Option<f64>) -> f64 {
        let declared = self.declared_headroom_gb(committed_gb, budget_gb);
        match available_gb {
            Some(available) => declared.min(available - self.safety_margin_gb),
            None => declared,
        }
    }

    /// Does a Scene's total footprint plus the margin fit under its budget?
    pub fn fits(&self, footprint_gb: f64, budget_gb: f64) -> bool {
        footprint_gb + self.safety_margin_gb <= budget_gb
    }

    /// Should the watchdog shed a model at this much memory available to models?
    pub fn below_floor(&self, model_available_gb: f64) -> bool {
        model_available_gb < self.floor_gb
    }
}

fn unused_gb(r: &Reservation, used_gb: Option<f64>) -> f64 {
    (r.gb - used_gb.unwrap_or(0.0)).max(0.0)
}

/// `memory.current` of a cgroup v2 path under /sys/fs/cgroup, in GB. Only plain
/// relative paths are accepted; the registry must not be able to point this at an
/// arbitrary file.
fn cgroup_used_gb(path: &str) -> Option<f64> {
    let rel = std::path::Path::new(path);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return None;
    }
    let bytes: f64 = std::fs::read_to_string(
        std::path::Path::new("/sys/fs/cgroup")
            .join(rel)
            .join("memory.current"),
    )
    .ok()?
    .trim()
    .parse()
    .ok()?;
    Some(bytes / 1024.0 / 1024.0 / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 108 GB Scene budget on a box with plenty free — the ordinary case.
    const BUDGET: f64 = 108.0;
    const U: Pool = Pool::unified();

    #[test]
    fn admits_when_both_declared_and_observed_ceilings_allow_it() {
        assert!(U.can_admit(66.0, 9.0, BUDGET, Some(100.0)));
    }

    #[test]
    fn refuses_when_the_declared_budget_is_exceeded() {
        // 66 + 40 + 8 margin = 114 > 108, even though the box has memory free.
        assert!(!U.can_admit(66.0, 40.0, BUDGET, Some(100.0)));
    }

    #[test]
    fn refuses_when_the_box_is_out_of_memory_even_though_the_paperwork_agrees() {
        // This is the case the declared check alone could never catch: nothing is
        // registered as committed, so the budget says there is 100 GB of room, but
        // MemAvailable says 20 GB. Something outside Lumbridge Computer is holding the pool.
        assert!(
            U.can_admit(66.0, 0.0, BUDGET, None),
            "declared check passes"
        );
        assert!(!U.can_admit(66.0, 0.0, BUDGET, Some(20.0)));
    }

    #[test]
    fn the_safety_margin_is_enforced_against_observed_memory_too() {
        // 66 GB model with exactly 66 GB free is a refusal: the margin has to fit.
        assert!(!U.can_admit(66.0, 0.0, BUDGET, Some(66.0)));
        assert!(!U.can_admit(66.0, 0.0, BUDGET, Some(66.0 + SAFETY_MARGIN_GB - 0.1)));
        assert!(U.can_admit(66.0, 0.0, BUDGET, Some(66.0 + SAFETY_MARGIN_GB)));
    }

    #[test]
    fn failed_sensing_falls_back_to_the_declared_ceiling_rather_than_refusing_everything() {
        assert!(U.can_admit(66.0, 9.0, BUDGET, None));
        // ...but it must not become a way to bypass the declared budget.
        assert!(!U.can_admit(66.0, 40.0, BUDGET, None));
    }

    #[test]
    fn a_zero_reading_refuses_everything_rather_than_admitting_everything() {
        // parse_meminfo yields 0.0 for an unparseable /proc. That has to read as
        // "no memory", not as "no constraint".
        assert!(!U.can_admit(1.0, 0.0, BUDGET, Some(0.0)));
    }

    #[test]
    fn headroom_never_exceeds_what_the_machine_actually_has() {
        // The shape of a real regression: a generous declared budget on a smaller box. Reporting
        // budget-minus-committed here promises room the next admission will refuse, and both the
        // MCP tool and the HTTP API serve this number to callers that cannot check it themselves.
        let box_available = 55.8;
        let declared_budget = 100.0;
        let reported = U.headroom_gb(0.0, declared_budget, Some(box_available));
        assert!(
            reported <= box_available,
            "reported {reported} GB of headroom on a box with {box_available} GB free",
        );
        assert!(U.can_admit(reported, 0.0, declared_budget, Some(box_available)));
    }

    #[test]
    fn headroom_reports_the_binding_constraint_not_the_generous_one() {
        // Declared says 33 GB spare; the box says 12 GB spare minus margin.
        assert_eq!(U.headroom_gb(67.0, BUDGET, None), 33.0);
        assert_eq!(U.headroom_gb(67.0, BUDGET, Some(12.0)), 4.0);
        // And the declared ceiling still wins when it is the tighter of the two.
        assert_eq!(U.headroom_gb(100.0, BUDGET, Some(90.0)), 0.0);
    }

    fn server(gb: f64, cgroup: Option<&str>) -> Reservation {
        Reservation {
            gb,
            cgroup: cgroup.map(str::to_string),
            description: String::new(),
            pool: PoolId::Unified,
        }
    }

    #[test]
    fn only_the_unclaimed_part_of_a_reservation_is_held_back() {
        // The Server promised 8.5 GB and uses 1.3 today. MemAvailable already excludes the
        // 1.3; holding back the full 8.5 would count it twice.
        assert!((unused_gb(&server(8.5, None), Some(1.3)) - 7.2).abs() < 1e-9);
        // A tenant over its promise is not a credit to models.
        assert_eq!(unused_gb(&server(8.5, None), Some(9.0)), 0.0);
        // Unmeasurable (stopped, no cgroup, unreadable): keep all of its room.
        assert_eq!(unused_gb(&server(8.5, None), None), 8.5);
    }

    #[test]
    fn the_observed_ceiling_uses_the_real_size_and_the_declared_one_the_typed_size() {
        // A real case: embed declared 10 GB, really 8.3; 17.8 GB available to
        // models. The typed number misses `10 + 8` by 0.2 GB; the real one leaves 9.5.
        assert!(!U.can_admit(10.0, 91.0, 111.0, Some(17.8)));
        assert!(U.can_admit_observed(10.0, 8.3, 91.0, 111.0, Some(17.8)));
        // The margin itself is not negotiable: 8.3 + 8 > 16.
        assert!(!U.can_admit_observed(10.0, 8.3, 91.0, 111.0, Some(16.0)));
        // A model that really takes more than it declares is held to the real number.
        assert!(!U.can_admit_observed(10.0, 12.0, 91.0, 111.0, Some(19.0)));
        // The declared ceiling still uses the typed figure, whatever was measured.
        assert!(!U.can_admit_observed(13.0, 8.3, 91.0, 111.0, Some(40.0)));
        // Sensing failed: declared ceiling alone, as before.
        assert!(U.can_admit_observed(10.0, 8.3, 91.0, 111.0, None));
    }

    #[test]
    fn reservations_shrink_what_admission_sees() {
        // 20 GB free on the box, 7.2 GB of it still promised to the Server: a 10 GB model
        // with the 8 GB margin no longer fits, though it would have without the Server.
        let available = 20.0 - 7.2;
        assert!(U.can_admit(10.0, 0.0, BUDGET, Some(20.0)));
        assert!(!U.can_admit(10.0, 0.0, BUDGET, Some(available)));
    }

    #[test]
    fn budget_never_exceeds_the_machine_minus_reservations() {
        // The shipped Scene declares 125 GB on a 121.7 GB Spark.
        assert!((U.effective_budget_gb(125.0, Some(121.7), 0.0) - 121.7).abs() < 1e-9);
        assert!((U.effective_budget_gb(125.0, Some(121.7), 8.5) - 113.2).abs() < 1e-9);
        // A modest budget is left alone, and so is everything when sensing failed.
        assert_eq!(U.effective_budget_gb(100.0, Some(121.7), 8.5), 100.0);
        assert_eq!(U.effective_budget_gb(125.0, None, 8.5), 125.0);
    }

    #[test]
    fn a_reservation_cannot_point_outside_the_cgroup_tree() {
        assert_eq!(cgroup_used_gb("../../etc"), None);
        assert_eq!(cgroup_used_gb("/proc/self"), None);
        assert_eq!(cgroup_used_gb("lumbridge.slice/../../proc"), None);
        assert_eq!(
            cgroup_used_gb("lumbridge.slice/definitely-not-a-cgroup.slice"),
            None
        );
    }

    #[test]
    fn registry_reservations_parse_and_default_to_none() {
        let without: Registry =
            serde_yaml::from_str("apiVersion: lumbridge/v1\nmodels: {}\n").unwrap();
        assert_eq!(U.reserved_gb(&without), 0.0);
        assert_eq!(U.reserved_unused_gb(&without), 0.0);

        let with: Registry = serde_yaml::from_str(
            "apiVersion: lumbridge/v1\nmodels: {}\nreservations:\n  server:\n    gb: 8.5\n",
        )
        .unwrap();
        assert_eq!(U.reserved_gb(&with), 8.5);
        // No cgroup to measure: the whole reservation is held back.
        assert_eq!(U.reserved_unused_gb(&with), 8.5);
        assert_eq!(U.model_available_gb(&with, 30.0), 21.5);
    }

    #[test]
    fn the_watchdog_floor_is_a_strict_lower_bound_and_can_be_overridden() {
        assert!(U.below_floor(2.9));
        assert!(!U.below_floor(WATCHDOG_FLOOR_GB));
        assert!(U.with_floor(5.0).below_floor(4.0));
        assert_eq!(U.with_floor(5.0).safety_margin_gb, SAFETY_MARGIN_GB);
    }

    #[test]
    fn models_and_reservations_default_to_the_unified_pool() {
        let reg = crate::config::Registry::parse(
            "apiVersion: lumbridge/v1\nreservations:\n  s:\n    gb: 1\nmodels:\n  m:\n    name: m\n    footprint_gb: 1\n    serve:\n      kind: exec\n",
        )
        .unwrap();
        assert_eq!(reg.models["m"].pool, PoolId::Unified);
        assert_eq!(reg.reservations["s"].pool, PoolId::Unified);
        assert!(U.holds(&reg.models["m"]));
        assert_eq!(PoolId::Unified.pool(), U);
        assert_eq!(PoolId::Unified.to_string(), "unified");
    }

    #[test]
    fn an_explicit_unified_pool_loads_and_an_unknown_pool_is_a_load_error() {
        let model = |pool: &str| {
            format!(
                "apiVersion: lumbridge/v1\nmodels:\n  m:\n    name: m\n    footprint_gb: 1\n    pool: {pool}\n    serve:\n      kind: exec\n"
            )
        };
        assert!(crate::config::Registry::parse(&model("unified")).is_ok());
        let err = format!(
            "{:#}",
            crate::config::Registry::parse(&model("vram")).unwrap_err()
        );
        assert!(
            err.contains("unknown variant") && err.contains("vram"),
            "{err}"
        );
        // Through `extends:` resolution too, not only the plain-model path.
        let extended = "apiVersion: lumbridge/v1\nmodels:\n  base:\n    name: b\n    footprint_gb: 1\n    serve:\n      kind: exec\n  child:\n    extends: base\n    pool: vram\n";
        assert!(crate::config::Registry::parse(extended).is_err());
        let reservation =
            "apiVersion: lumbridge/v1\nmodels: {}\nreservations:\n  s:\n    gb: 1\n    pool: vram\n";
        assert!(crate::config::Registry::parse(reservation).is_err());
    }
}

/// Proof that introducing pools changed nothing: the admission math from before
/// the refactor, copied verbatim from `governor.rs`, must give the same verdicts
/// and the same headroom numbers as the pool for every model of every Scene in
/// the fixture box, across the memory states such a box actually sees.
#[cfg(test)]
mod equivalence {
    use super::*;
    use crate::config::{load_scenes, Registry, Scene};
    use std::path::Path;

    /// `governor.rs` before pools, admission half, unchanged.
    mod legacy {
        use crate::config::{Registry, Reservation};
        pub const DEFAULT_BUDGET_GB: f64 = 100.0;
        pub const SAFETY_MARGIN_GB: f64 = 8.0;
        pub const WATCHDOG_FLOOR_GB: f64 = 3.0;

        pub fn reserved_gb(reg: &Registry) -> f64 {
            reg.reservations.values().map(|r| r.gb).sum()
        }
        pub fn reserved_unused_gb(reg: &Registry, used: Option<f64>) -> f64 {
            reg.reservations.values().map(|r| unused_gb(r, used)).sum()
        }
        fn unused_gb(r: &Reservation, used_gb: Option<f64>) -> f64 {
            (r.gb - used_gb.unwrap_or(0.0)).max(0.0)
        }
        pub fn model_available_gb(reg: &Registry, available_gb: f64, used: Option<f64>) -> f64 {
            available_gb - reserved_unused_gb(reg, used)
        }
        pub fn effective_budget_gb(budget_gb: f64, total_gb: Option<f64>, reserved_gb: f64) -> f64 {
            match total_gb {
                Some(total) => budget_gb.min(total - reserved_gb),
                None => budget_gb,
            }
        }
        pub fn can_admit(
            add_gb: f64,
            committed_gb: f64,
            budget_gb: f64,
            available_gb: Option<f64>,
        ) -> bool {
            if committed_gb + add_gb + SAFETY_MARGIN_GB > budget_gb {
                return false;
            }
            match available_gb {
                Some(available) => add_gb + SAFETY_MARGIN_GB <= available,
                None => true,
            }
        }
        pub fn headroom_gb(committed_gb: f64, budget_gb: f64, available_gb: Option<f64>) -> f64 {
            let declared = budget_gb - committed_gb - SAFETY_MARGIN_GB;
            match available_gb {
                Some(available) => declared.min(available - SAFETY_MARGIN_GB),
                None => declared,
            }
        }
    }

    fn fixture() -> (Registry, Vec<Scene>) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/box");
        let registry = Registry::load(&root).unwrap();
        let scenes = load_scenes(&root).unwrap();
        assert!(scenes.len() >= 15, "fixture looks truncated");
        (registry, scenes)
    }

    /// The order activation starts a Scene's models in (lifecycle::plan_transition).
    fn start_order<'a>(registry: &'a Registry, scene: &'a Scene) -> Vec<&'a str> {
        let mut ids: Vec<&str> = scene.models.iter().map(String::as_str).collect();
        let listed = scene
            .activation
            .as_ref()
            .and_then(|a| a.order.as_deref())
            .is_some_and(|o| o == "listed");
        if !listed {
            ids.sort_by(|a, b| {
                registry.models[*a]
                    .footprint_gb
                    .total_cmp(&registry.models[*b].footprint_gb)
            });
        }
        ids
    }

    // MemTotal and MemAvailable as a busy 128 GB box reports them, around them the
    // tight and the empty box, and `None` for failed sensing.
    const TOTALS: [Option<f64>; 2] = [Some(127_600_816.0 / 1048576.0), None];
    const AVAILABLE: [Option<f64>; 6] = [
        Some(23_963_520.0 / 1048576.0),
        Some(0.0),
        Some(11.0),
        Some(60.0),
        Some(118.5),
        None,
    ];
    // The Supabase slice: unmeasurable (whole 8.5 GB held), 4.09 GB in use, over its promise.
    const SLICE_USED: [Option<f64>; 3] = [None, Some(4_395_565_056.0 / 1073741824.0), Some(9.0)];

    #[test]
    fn the_unified_pool_carries_the_governors_old_numbers() {
        let u = Pool::unified();
        assert_eq!(u.budget_gb, legacy::DEFAULT_BUDGET_GB);
        assert_eq!(u.safety_margin_gb, legacy::SAFETY_MARGIN_GB);
        assert_eq!(u.floor_gb, legacy::WATCHDOG_FLOOR_GB);
    }

    #[test]
    fn every_fixture_model_and_reservation_lands_in_the_unified_pool() {
        let (registry, _) = fixture();
        assert!(registry
            .models
            .values()
            .all(|m| m.pool.pool() == Pool::unified()));
        assert!(registry
            .reservations
            .values()
            .all(|r| r.pool == PoolId::Unified));
        let u = Pool::unified();
        assert_eq!(u.reserved_gb(&registry), legacy::reserved_gb(&registry));
    }

    #[test]
    fn fixture_scene_fit_verdicts_are_unchanged() {
        let (registry, scenes) = fixture();
        let mut fits = 0;
        for scene in &scenes {
            let budget = scene.budget_gb.unwrap_or(legacy::DEFAULT_BUDGET_GB);
            let footprint: f64 = scene
                .models
                .iter()
                .filter_map(|id| registry.models.get(id))
                .map(|m| m.footprint_gb)
                .sum();
            let resolvable = scene
                .models
                .iter()
                .all(|id| registry.models.contains_key(id))
                && scene.check_ports(&registry).is_ok();
            let before = resolvable && footprint + legacy::SAFETY_MARGIN_GB <= budget;
            let after = crate::lifecycle::validate_scene(&registry, scene).is_ok();
            assert_eq!(before, after, "scene {}", scene.metadata.name);
            fits += usize::from(after);
        }
        // Not a vacuous comparison: most shipped Scenes validate.
        assert!(fits >= scenes.len() / 2, "{fits} of {}", scenes.len());
    }

    #[test]
    fn fixture_admission_verdicts_and_headroom_are_unchanged_for_every_scene_start() {
        let (registry, scenes) = fixture();
        let reserved = legacy::reserved_gb(&registry);
        let (mut compared, mut admitted, mut refused) = (0usize, 0usize, 0usize);
        for scene in &scenes {
            let declared = scene.budget_gb.unwrap_or(legacy::DEFAULT_BUDGET_GB);
            for total in TOTALS {
                let before_budget = legacy::effective_budget_gb(declared, total, reserved);
                let pool = Pool::unified();
                let after_budget = pool.effective_budget_gb(
                    pool.scene_budget_gb(scene.budget_gb),
                    total,
                    pool.reserved_gb(&registry),
                );
                assert_eq!(before_budget.to_bits(), after_budget.to_bits());
                for available in AVAILABLE {
                    for used in SLICE_USED {
                        let before_avail =
                            available.map(|a| legacy::model_available_gb(&registry, a, used));
                        // The pool reads the slice itself; feed it the same reading.
                        let unused = pool.reserved_unused_with(&registry, |_| used);
                        let after_avail = available.map(|a| a - unused);
                        assert_eq!(
                            before_avail.map(f64::to_bits),
                            after_avail.map(f64::to_bits)
                        );

                        let mut committed = 0.0;
                        for id in start_order(&registry, scene) {
                            let Some(model) = registry.models.get(id) else {
                                continue;
                            };
                            let pool = model.pool.pool();
                            let before = legacy::can_admit(
                                model.footprint_gb,
                                committed,
                                before_budget,
                                before_avail,
                            );
                            let after = pool.can_admit(
                                model.footprint_gb,
                                committed,
                                after_budget,
                                after_avail,
                            );
                            assert_eq!(
                                before, after,
                                "{} / {id}: total {total:?} available {available:?} slice {used:?}",
                                scene.metadata.name
                            );
                            let h_before =
                                legacy::headroom_gb(committed, before_budget, before_avail);
                            let h_after = pool.headroom_gb(committed, after_budget, after_avail);
                            assert_eq!(h_before.to_bits(), h_after.to_bits());
                            compared += 1;
                            if after {
                                admitted += 1;
                                committed += model.footprint_gb;
                            } else {
                                refused += 1;
                            }
                        }
                        // Status reports headroom with nothing running too.
                        assert_eq!(
                            legacy::headroom_gb(0.0, before_budget, before_avail).to_bits(),
                            pool.headroom_gb(0.0, after_budget, after_avail).to_bits()
                        );
                    }
                }
            }
        }
        // Both verdicts occur, so the comparison covers both branches.
        assert!(
            compared > 1000 && admitted > 100 && refused > 100,
            "{compared} {admitted} {refused}"
        );
    }

    #[test]
    fn the_watchdog_trips_at_the_same_readings() {
        let (registry, _) = fixture();
        let pool = Pool::unified();
        for available in [0.0, 2.99, 3.0, 3.01, 11.4, 11.6, 23.0] {
            for used in SLICE_USED {
                let before = legacy::model_available_gb(&registry, available, used);
                let after = available - pool.reserved_unused_with(&registry, |_| used);
                assert_eq!(before.to_bits(), after.to_bits());
                assert_eq!(before < legacy::WATCHDOG_FLOOR_GB, pool.below_floor(after));
            }
        }
    }
}
