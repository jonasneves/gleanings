# Gleanings

What the attention economy's harvesters left behind.

Gleanings is an automated research pipeline that surfaces AI breakthroughs which are technically significant but under-covered. Four AI agents research in parallel every week, a synthesizer finds cross-cutting themes, and the results are published as a static site on GitHub Pages.

## What it finds

**Research track** — Papers and findings that challenge assumptions the field is operating on. Negative results, cross-domain discoveries, mechanistic insights. Ranked by disruption to current practice, not by benchmark numbers or lab brand.

**Practitioner track** — Things people discovered while actually using Claude, ChatGPT, and other AI tools. Shared once in a Discord server or Reddit thread and then disappeared. The people who found them often didn't realize how significant the discovery was.

## How it works

```
agents/
  reasoning_inference.md       → reasoning, CoT faithfulness, test-time compute
  architecture_training.md     → architectures, training paradigms, data efficiency
  agents_science.md            → agents, multimodal, AI for science
  practitioner_discoveries.md  → things found in the wild
  synthesizer.md               → cross-cutting themes, significance ranking
```

All four research agents run in parallel. The synthesizer deduplicates, identifies convergent themes across agents, and ranks by significance. Output is a self-contained HTML page.

## Running locally

```bash
pip install -r requirements.txt

# With GitHub Models (if you have Copilot)
export GITHUB_TOKEN=your_token
python run.py

# With Anthropic API
export ANTHROPIC_API_KEY=your_key
python run.py --provider anthropic

# Override model
python run.py --model claude-opus-4-6
```

## Automated publishing

The GitHub Actions workflow runs every Monday at 9am UTC and deploys to GitHub Pages. You can also trigger it manually from the Actions tab.

To set up: add `GITHUB_TOKEN` (with `models:read` scope) or `ANTHROPIC_API_KEY` as a repository secret.

## Adding a research agent

1. Create `agents/your_topic.md` following the format of existing agent files
2. Add the filename to the `RESEARCH_AGENTS` list in `run.py`
3. The synthesizer automatically includes its output

## Design lens

Findings are evaluated through the `ai-research-signal` lens: coverage is driven by announcement cadence and lab brand, not by actual implications. The agents prioritize negative results, cross-domain findings, and mechanistic insights over benchmark headlines.

## License

MIT
