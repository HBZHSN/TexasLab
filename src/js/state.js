export function createInitialState() {
  return {
    players: 6,
    street: "flop",
    target: "river",
    // For a bounded [0, 1] equity estimate, 2,000 samples give a
    // distribution-free 95% Hoeffding radius of about 3.1 percentage points.
    // Keep larger manual verification tiers in the UI.
    iterations: 2000,
    slots: {
      h0: null,
      h1: null,
      b0: null,
      b1: null,
      b2: null,
      b3: null,
      b4: null,
    },
    activeSlot: "h0",
    lastRows: [],
    busy: false,
  };
}
