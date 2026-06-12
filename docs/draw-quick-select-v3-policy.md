# Draw Quick Select V3 Policy

Goal: cut where a human can read a natural image boundary. A single stroke does
not need to be perfect, but add/erase strokes must compose predictably into the
desired mask.

## Controls

### Brush Size

Brush size defines the user's selected-side seed for one stroke. It may protect
the stroke interior from collapsing, but it must not by itself create outside
growth.

### Radius

Radius defines the search distance for the current stroke. It bounds where the
solver may look for a boundary. Increasing radius must not make a featureless
stroke inflate unless an accepted image edge or accepted same-side fill justifies
the added pixels.

### EdgeLock

EdgeLock is edge sensitivity. Internally it resolves to an effective 0-100
policy value:

* lower values: accept only strong, crisp ridges
* higher values: accept weaker or more diffuse ridges

EdgeLock affects edge confidence thresholds and graph edge cost only. It should
not directly change boundary-side bias, alpha softness, or blind dilation.

Wide or thin strokes can need extra side separation inside the brush body:

* thin elongated strokes may lower the side-split threshold so top/bottom edges
  can trim the brush footprint
* subtle broad bright dabs may apply a weak inside colour BG prior when the
  opposite side is only slightly different from the seed

These corrections are local to the stroke. They do not change Radius and they do
not let support grow outside the candidate band.

### Boundary Bias

Boundary Bias is a pixel offset around an already accepted edge. It decides which
side of the accepted boundary ridge receives the edge pixels. It must not change
edge sensitivity or foreground/background colour membership. Soft alpha may use
the same UI value, but only to change edge opacity, not binary support reach.

### Alpha Softness

Alpha softness changes only the final alpha inside the selected support. It must
not alter the binary support mask.

## Per-Stroke Invariants

Each add stroke is solved independently in V3. The following values are local to
that stroke and should be exposed in debug planes:

* effective EdgeLock
* ridge threshold
* restore threshold
* side-split threshold
* outside-keep threshold
* boundary bias in pixels
* local colour model

Adding a second stroke must not recompute the first stroke's edge thresholds or
colour model. Final support is composed by unioning add strokes, then applying
erase semantics.

## Debug Planes

The solver writes policy planes so a captured `qs_input_*.npz` can explain why a
boundary moved:

* `edge_policy_ridge_threshold`
* `edge_policy_restore_threshold`
* `edge_policy_side_threshold`
* `edge_policy_outside_keep_threshold`
* `boundary_bias_px`

These names are part of the debugging contract. If a future change makes a UI
control feel wrong, first check which plane changed and whether it matches the
control's declared scope above.
