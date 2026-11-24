//! Withholds `send_via` from registries that own no channel map.

use crate::tools::Tool;
use crate::tools::scoped::ScopedToolRegistry;

/// The name `SendViaTool` registers under.
const SEND_VIA: &str = "send_via";

/// Drops `send_via` from the registry of an independent delegate target or a
/// nested SOP step.
///
/// Both builders keep only the assembled registry and discard the late-bound
/// channel handles that registry construction returns, so the tool's channel
/// map stays empty. Every call that names a `target` then fails, and an
/// immediate send requires one. That leaves a modality-only routing call as the
/// single form that can succeed: when the call runs on a channel turn's task,
/// it writes into that turn's `TURN_ROUTING` handle and restyles the turn's own
/// reply, from an agent running under its own risk profile.
///
/// Seeding these registries with the live channel map is what would make their
/// own `send_via` real; until then the tool has nothing it could honour. A
/// bounded delegate is different: it reuses the caller's tools, and with them
/// the caller's `SendViaTool` and its real channel map.
pub(crate) fn withhold_send_via(registry: &mut ScopedToolRegistry) {
    registry.retain(|tool| tool.name() != SEND_VIA);
}
