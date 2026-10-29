# AI assistants (Agent Skills)

The repository ships an [Agent Skill](https://agentskills.io) for SWEEP in
`skills/sweep-solver/`. A skill is a folder of Markdown instructions and
helper scripts that an AI coding assistant loads when a task needs it. With
it installed, Claude Code, Claude.ai, Codex and other assistants that read
the Agent Skills format write and debug `sweep` code with the library's real
API instead of guessing it: which equation and models to use, `impl='c'` vs
`impl='eager'`, boundaries and free surface, gradient memory options, imaging,
SEM, multi-GPU, the CLI and `sweep.datasets`, and the pitfalls that fail
silently.

The skill is written for the current release (0.3.6). It is not part of the
wheel; take it from the repository.

## Contents

| Path | What it holds |
|---|---|
| `SKILL.md` | Entry point: install, the minimal forward + gradient loop, the equation table, backends, memory options, a list of pitfalls, and pointers to the references |
| `references/equations.md` | Every equation: models, default source and receiver fields, compiled support, free surface |
| `references/propagators.md` | `PropTorch` / `PropJax` arguments, options objects, shapes and units |
| `references/boundaries.md` | CPML types, per-edge free surface, topography |
| `references/gradients.md` | Which gradients each backend returns, custom gradients, checkpointing and boundary saving |
| `references/imaging.md` | RTM, Born modelling and LSRTM (acoustic and elastic), RWI, ADCIG |
| `references/sem.md` | `PropSEM`, meshes and SEM equations |
| `references/parallel.md` | Shot parallelism, source encoding, domain decomposition (`ModelParallel`), JAX |
| `references/cli-datasets.md` | The `sweep` CLI and `sweep.datasets` |
| `references/api-signatures.md` | Signatures of the public classes, generated from the code |
| `scripts/check_env.py` | Reports whether an environment can run `sweep` and on which backend |
| `scripts/stability_check.py` | Checks a grid spacing and time step against the CFL bound of a scheme |

The assistant reads `SKILL.md` first and opens a reference file only when the
task needs it.

## Get the folder

Clone only the skill:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/DeepWave-KAUST/sweep.git sweep-skill
git -C sweep-skill sparse-checkout set skills/sweep-solver
```

The folder is then `sweep-skill/skills/sweep-solver`. A full clone of the
repository works as well.

## Install

Copy the folder, keeping its name `sweep-solver`, to where your assistant
looks for skills.

**Claude Code**: for all your projects, or only for one project:

```bash
mkdir -p ~/.claude/skills && cp -r sweep-skill/skills/sweep-solver ~/.claude/skills/
mkdir -p .claude/skills && cp -r sweep-skill/skills/sweep-solver .claude/skills/   # run in the project
```

**Codex**: the same, with `~/.agents/skills/` or the project's
`.agents/skills/`.

**Claude.ai**: zip the folder and upload the zip in the skills section of the
settings.

```bash
cd sweep-skill/skills && zip -r sweep-solver.zip sweep-solver
```

**Other assistants**: give them `SKILL.md` and let them open the reference
files it points to.

## Use

The assistant decides from the skill's description when to load it: any task
that imports `sweep`, mentions `PropTorch` or `sweep-solver`, or asks for
finite-difference seismic modelling or inversion. You can also ask for it by
name, for example "use the sweep-solver skill to write a 2-D elastic FWI on
Marmousi".

Before running generated code on a new machine, the assistant can check the
environment with

```bash
python ~/.claude/skills/sweep-solver/scripts/check_env.py
```

## Keeping it correct

Every code block in the skill runs in the test suite:
`test/test_skill_snippets.py` runs each Markdown file's blocks in a fresh
working directory with `skills/run_snippets.py`, so an API change that breaks
an example fails the tests. To check the skill by hand:

```bash
python skills/run_snippets.py skills/sweep-solver
```

After a change to a public signature, regenerate the signature reference:

```bash
cd skills/sweep-solver/scripts && python gen_api_signatures.py -o ../references/api-signatures.md
```
