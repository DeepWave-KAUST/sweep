# CLI

SWEEP provides a small CLI for inspecting available equations.

## List Equations

```bash
sweep list equations
```

This shows:

- Equation name
- Required model parameters
- Whether torch binding support exists
- Whether torch binding is available in the current environment

## Show Equation Details

```bash
sweep show Acoustic
```

This command displays:

- Wavefields
- Needed models
- Torch binding support
- Torch binding availability

## TODO

- Add CLI output examples
- Add more subcommands if the interface grows
