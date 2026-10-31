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

The skill is written for the current release. From sweep-solver 0.3.7 the
wheel carries it and `sweep skill install` puts it where your assistant
reads it; with 0.3.6, take it from the repository.

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

## Install

Run the command on the machine where the assistant runs (for a session on a
cluster, on the cluster):

```bash
sweep skill install
```

It copies the skill to `~/.claude/skills/sweep-solver/` for Claude Code and
to `~/.agents/skills/sweep-solver/` for Codex, for each of the two it finds
for your user (its `~/.claude`, `~/.agents` or `~/.codex` folder, or its
command on `PATH`). Run it again after upgrading `sweep-solver` to update the
copy. An assistant reads skills from these folders and nothing is registered,
so a skill copied before the assistant is installed is picked up once it is;
a session that is already running may need a restart.

| Command | What it does |
|---|---|
| `sweep skill install` | install for the assistants found; with none found, install nothing and list the options |
| `sweep skill install --for claude` / `--for codex` / `--for all` | install for those, found or not |
| `sweep skill install --project [DIR]` | install into `DIR/.claude/skills/` or `DIR/.agents/skills/` (default: the current directory), which is what a cloud session working on that repository sees |
| `sweep skill zip [-o FILE]` | write `sweep-solver.zip` for **Claude.ai**: upload it in the skills section of the settings |
| `sweep skill path` | print where the installed package keeps the skill |

An existing folder is replaced only when it is this skill; any other folder
of that name, or a symlink, is left alone with an error.

**From the repository** (sweep-solver 0.3.6, or other assistants): clone
only the skill and copy the folder, keeping its name `sweep-solver`, to one
of the folders above. Other assistants can be given `SKILL.md` and left to
open the reference files it points to.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/DeepWave-KAUST/sweep.git sweep-skill
git -C sweep-skill sparse-checkout set skills/sweep-solver
mkdir -p ~/.claude/skills && cp -r sweep-skill/skills/sweep-solver ~/.claude/skills/
```

## Use

The assistant decides from the skill's description when to load it: any task
that imports `sweep`, mentions `PropTorch` or `sweep-solver`, or asks for
finite-difference seismic modelling or inversion. You can also ask for it by
name, for example "use the sweep-solver skill to write a 2-D elastic FWI on
Marmousi".

Before running generated code on a new machine, the assistant can check the
environment with

```bash
python "$(sweep skill path)"/scripts/check_env.py
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
