# gleanings

Research pipeline for identifying under-covered AI breakthroughs. Runs a team of parallel agents, synthesizes findings, and publishes to HTML.

## Active lenses

- `ai-research-signal` — primary lens for this project; see `~/.claude/lenses/ai-research-signal.md`
- `signal-to-noise` — applied during synthesis and HTML generation
- `criterion` — coverage ≠ significance; do not let upvote/citation count stand in for actual importance
- `voice` — synthesis prose must arrive at the specific before the general; no AI-sounding structure

## Project structure

```
agents/           Agent prompt files (markdown). Each defines one research thread.
lenses/           Local copy of the domain lens for reference.
output/           Generated HTML reports. Git-ignored.
run.py            Orchestrator — runs agents in parallel, synthesizes, writes HTML.
```

## Running

```bash
python run.py
# Opens output/index.html when done
```

Requires `ANTHROPIC_API_KEY` in environment.

## Agent roles

| File | Focus |
|---|---|
| `agents/reasoning_inference.md` | Reasoning, CoT faithfulness, test-time compute, mechanistic interpretability |
| `agents/architecture_training.md` | Novel architectures, training paradigms, data efficiency, self-improvement |
| `agents/agents_science.md` | AI agents, memory, multimodal, AI for science (bio/chem/math/physics) |
| `agents/practitioner_discoveries.md` | Breakthroughs from people actually using Claude/AI — Reddit, HN, Discord, Twitter |
| `agents/synthesizer.md` | Cross-cutting themes, convergent findings, significance ranking |

The practitioner agent runs in parallel with the research agents but feeds a separate HTML section ("In the wild") rather than the synthesizer. Its output schema is different — no arXiv IDs, discoverer type, independent confirmation count.

## Lens application during runs

The `ai-research-signal` lens governs what each agent looks for. Agents must:
- Prioritize negative results and counterintuitive findings over positive capability claims
- Flag cross-domain findings explicitly
- Evaluate significance by disruption to current practice, not by benchmark numbers
- Separate mechanism findings from benchmark findings
- Note HN upvote counts and blog post presence as coverage signals (not quality signals)

## Adding a new agent

1. Create `agents/your_topic.md` following the format of existing agent files
2. Add it to the `RESEARCH_AGENTS` list in `run.py`
3. The synthesizer prompt will automatically include its output

## Output format

`output/index.html` — single self-contained file, no external dependencies.
Each run overwrites the previous output. To keep a run, copy with a datestamp.
