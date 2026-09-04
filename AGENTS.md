# AGENTS.md

Guidance for AI agents working on the timewizard repo. The CLAUDE.md
symlink resolves to this file. Read it top to bottom on first session.

## Project context

time-wizard fine-tunes LFM2.5-VL-450M to read the time from photographs
of analog clocks. The benchmark is 200 held-out photos from It's About
Time (Yang and Zisserman, 2022), scored within one minute. The goal is
a 450M parameter model that beats frontier models on those 200 photos.

The repo is Python only and deliberately small. Photos and labels come
from the itsabouttime CSVs and the COCO and OpenImages images. Training
uses TRL's SFTTrainer. Frontier baselines run through pydantic-ai. The
`timewizard` package holds only the glue: the task definition and metric,
the photo pipeline, the conversation builder, the training entry point,
the reader, and the bench runner.

## Repository layout

```
timewizard/    reading, photos, data, train, reader, bench, figures
benchmark/     frozen splits as ids and labels, plus the dataset card
model/         the model card and its figures
sky/           SkyPilot task for the Nebius GPU run
RESULTS.md     every score, frontier and ours
tests/         pytest suite
runs/          checkpoints and logs, gitignored except runs/bench/ replies
data/          downloaded crops, gitignored
justfile       task runner
```

Training runs on a Nebius GPU through SkyPilot, defined by
`sky/train.yaml`. Change a training flag through `--env TRAIN_ARGS`
rather than by editing the file, so the committed task stays the
reproducible one. README.md holds every instruction and RESULTS.md
every score. `benchmark/README.md` is the dataset card that
`timewizard.photos card` uploads to the Hugging Face dataset.
`model/README.md` is the model card that `just model-card` uploads to the
model repo. Put new prose in one of those four rather than adding a file
beside the code.

## Test data hygiene

`benchmark/photos_test.json` is evaluation data. Only `timewizard.bench`
reads it, and only with `--split test` for a final model. Training reads
`photos_train.json`. Hyperparameters are chosen on `photos_dev.json`.
Selection against a test set leaks the test set.

`timewizard.photos build` froze the splits from seed 0. Changing the seed,
the deduplication threshold, or the crop margin creates a new benchmark
and invalidates every number reported so far. Do that only with a version
bump and a note in the README. Push the rebuilt crops afterwards, because
every machine pulls them from the Hugging Face dataset.

## Quality gate

Run `just check` before declaring work done. The recipe runs `ruff
format --check`, `ruff check`, `ty check`, and `pytest`, read-only,
the way CI would. CLIs are tyro dataclasses. Structured answers are
pydantic models. If the gate does not pass locally, the work is not
done.

## Writing prose

The following style rules apply to all prose in the repo: README,
docs, reports, commit message bodies, code comments. They are the
maintainer's stated preference and must be honored.

### Hard rules

- **No em-dashes.** The character does not appear in prose. If you
  would normally use an em-dash, split into two sentences or use a
  comma. The same goes for en-dashes in prose.
- **No semicolons in prose.** Use a period and start a new sentence.
- **No unnecessary parentheses.** A parenthetical aside that pauses
  the reader for a thought you could have put in its own sentence
  should go in its own sentence. Parens are fine for genuine
  clarifications, such as an abbreviation on first use, but not as a
  substitute for a comma or period.
- **No ASCII diagrams.** Describe relationships in prose. A single
  inline arrow like `plan -> answer` is fine, boxes and arrows are
  not.
- **No emoji** unless the user explicitly asks for them.
- **No vague back-references.** Do not open a sentence with "This",
  "That", "These", "Those", "Their", or "It" pointing at a noun from
  an earlier sentence. Name the noun again. The reader should never
  have to look backward to resolve what a pronoun stands for.
- **No negation-contrast framing.** State what the code or the text
  does. "The reader outputs the floor of the minute" is right. "The
  reader does not round, it floors" is wrong.
- **No clause joins.** A comma followed by "and", "so", "but", or
  "which" glues two sentences together. Write two sentences. "The
  runner writes each reply as it arrives, so a crash costs nothing" is
  wrong. "The runner writes each reply as it arrives. A crash therefore
  costs nothing" is right.
- **Active voice.** Name the actor and give it the verb. "The runner
  writes the score" is right. "The score is written" is wrong.
- **One idea per sentence.** A sentence that lists four actions of a
  pipeline belongs in a numbered list. Each step gets its own line.

### Soft rules

- Write short, direct sentences. A sentence with more than one comma is
  usually two sentences.
- Define every term the reader may not hold. Name the thing a perceptual
  hash gives you. Say what an ablation removes. A term left undefined is
  jargon.
- Lead with the noun, not the qualifier. "The renderer draws the
  bezel first" beats "Before anything else, the renderer draws the
  bezel."
- Define jargon on first use, even if you think the reader knows it.
- Do not write in fragments or in a punchy, aphoristic style. Short
  clipped clauses strung together read like a parable. "No GPU, no
  cost, runs on a laptop" is wrong. "The renderer runs on a CPU, so
  dataset generation costs nothing beyond wall clock" is right.
- Every sentence names a concrete actor or object. Slogans and stock
  metaphors are out.

## Writing comments

The rules under "Writing prose" apply, plus:

- **Default to writing no comment.** A well-named identifier and a
  short function explain themselves. Only comment when the why is
  non-obvious: a hidden constraint, a subtle invariant, a workaround
  for a specific upstream bug, behavior that would surprise a reader.
- **Don't describe what the code does.** The code does that.
- **Say what is done and why.** Frame a comment around the present
  behavior and its reason. Name the road taken.
- **Don't reference the past.** "Renamed from X" and "formerly Y" rot.
  Comments describe the present state.
- **Don't reference callers or PRs.** They rot as the code evolves.
- **Don't write multi-line comment banners.** One short comment per
  declaration.

## Writing documentation and reports

- **Explain every code block.** Never drop a command or snippet
  without saying what it does and what every meaningful flag means.
  Show output too, and say what its columns or fields mean. Put that
  explanation in prose around the block. Never append a comment to a
  command inside one, which cannot be copied and read at once.
- **Headings name content.** "Dataset generation" is good. "Now we
  generate the data" narrates the act instead of naming the subject.
- **Do not over-chunk.** A heading breaks the reader's flow. Add one
  only where a genuinely new section begins.
- **Cut filler.** Remove words that earn nothing.

## Writing tests

Use `pytest`. Test the happy path for every exported function, every
validation branch that raises, and boundary cases for numeric inputs:
the wrap from 12:59 to 1:00, the 12 o'clock hour, replies with extra
text around the JSON. Parametrize with
`@pytest.mark.parametrize` for multiple cases of the same shape,
standalone `test_<scenario>` functions otherwise. Prefer plain
`assert`. Use hand-written fakes over mock libraries where practical.
Tests never load a model and never touch the network. Test the metric
on hand-written times and the renderer through its labels.

## Commit messages

Conventional commits, one sentence each, no body unless absolutely
necessary.

```
feat: resume a scoring run from its replies file
fix: wrap the circular error at twelve hours
docs: describe the dev split
chore: pin tenacity
```

Rules:

- One sentence subject. Pick a tense and be consistent.
- Lowercase the type and the first word after the colon, unless it is
  a proper noun or acronym.
- No commit body unless the reason cannot fit in the subject. No "Test
  plan" or "Summary" boilerplate.
- **No `Co-Authored-By: Claude` trailer.** Ever, even when commits are
  authorized in advance. No AI attribution in PR bodies either.
- Do not amend or rewrite published commits without explicit user
  consent. Force-push only with `--force-with-lease`, only on a
  feature branch, and only after confirming.

## Git practices

- Use whatever git identity the user has configured. Never pass
  `-c user.email` or `-c user.name`.
- Don't push without explicit authorization. Ask before every commit
  and every push, each time, even after a prior yes.
- Before any destructive operation, confirm with the user.
- Submodules pin an upstream commit. After changing a submodule's
  checked-out commit, commit the new pointer with a `chore:` message
  that says what moved and why.

## Python style

Type everything, fail loudly, prefer the standard library, and let the
tooling enforce the rest.

### Tooling

- **uv** for environments and dependencies. Use `uv add`, `uv lock`,
  `uv run`.
- **ruff** for both linting and formatting.
- **ty** for type checking.
- **just** for task running: `setup`, `photos`, `train`, `bench`,
  `fmt`, and a `check` that runs the read-only set.

### Project layout and dependencies

- Runtime dependencies go in `[project].dependencies`. Dev tools go in
  `[dependency-groups].dev` per PEP 735. Torch, transformers, TRL, and
  PEFT go in the `train` group, pydantic-ai in `baselines`. All groups
  install by default so the type checker sees every module.
- Pin every direct dependency to an exact version with `==` and commit
  `uv.lock`. The lockfile is never hand-edited.

### Types

Type every function signature, both parameters and return.

- Put `from __future__ import annotations` at the top of every module.
- Use built-in generics, `list[int]` and `dict[str, T]`. Use
  `X | None`.
- Model structured data with a pydantic model: `Time`, `Photo`, `Score`,
  `Report`, and every CLI config. A signature like `list[dict[str, Any]]`
  is the warning sign: the rows have a shape, so give the shape a model.
  tyro builds the CLI from a pydantic config and enforces its validators,
  and `TypeAdapter` reads and writes the JSON files.

### Imports

Every import goes at the top of the module. Never import inside a
function. A function-level import hides a dependency from the reader and
the tooling, and it moves an ImportError from startup to call time. Every
dependency group installs by default, so nothing here needs a deferred
import.

### Naming and errors

- `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_SNAKE` for module constants. A leading underscore marks a
  name module-private.
- Raise exceptions. Return values carry results only.
- Catch narrowly. A broad `except Exception` belongs only at a
  top-level boundary where you log and carry on. Use
  `raise ... from err` to preserve the cause.

### Don't reinvent the wheel

Prefer the standard library or a well-maintained dependency over code
you write yourself. TRL trains. pydantic-ai talks to API models.
imagehash deduplicates. tyro parses arguments. tenacity retries. tqdm
reports progress. Write your own implementation only when nothing fits.

## Long-running commands

Run anything that can outlast a few seconds in the background. That covers
`sky launch`, `sky down`, `sky status`, a training run, a scoring run, an
upload, and an image download. Watch its output file with a monitor rather
than polling. A foreground command that needs a timeout is a command that
belonged in the background.

## Working with the user

- Local, reversible actions need no preamble. Hard-to-reverse actions
  need explicit confirmation each time.
- Default to terse. Lead with the result.
- One or two sentence end-of-turn summary: what shipped and what is
  next.
- Match the scope of changes to what the user asked. A bug fix does
  not get a free refactor.
- When you spot a side-effect the user did not ask for, name it and
  ask before doing it.

## When in doubt

Re-read this document, then the most recent changes that touched the
same area. Match the existing patterns rather than introducing a new
variation.
