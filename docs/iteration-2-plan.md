# Iteration 2 — Document classification and the MCP server

> **Status.** Not built. Iteration 1 is complete on `main`.
>
> **Ordering change.** The iteration-1 plan sketched the watcher as iteration 2
> and the MCP server as iteration 3. This document supersedes that: the
> classifier and the MCP server come first, and the watcher moves after them.
> The reasoning is in "Why this before the watcher" below.

## Context

The index can already find things by meaning. What it cannot do is distinguish
*what a document is for*, and that turns out to matter for the case this tool
exists to serve.

The motivating scenario: an agent picks up a work item on a greenfield area of a
project. There is no existing code to imitate. What should guide it are the
technical specifications, architecture documents and conventions that say how
things are *supposed* to be built. Today a search returns whatever is
semantically nearest — a changelog entry, a meeting note and a spec are
indistinguishable to the index, because all three are `FileKind.MARKDOWN`.

We already shipped a workaround for a narrow slice of this problem, which is the
clearest evidence the gap is real. From `config/workspace.example.yaml`:

```yaml
instruction: >
  Prioritize implementation code over tests, fixtures, and generated files.
```

That is a natural-language plea to a reranker to do what a typed field should do
deterministically. It works sometimes, cannot be filtered on, cannot be counted,
and cannot be checked.

## The distinction: format versus purpose

`FileKind` answers **"how do I chunk this"** — code, markdown, text, image. It
selects a chunking strategy and nothing else.

The new axis answers **"what role does this document play"** — is it telling me
how things must be built, describing what exists, or recording what happened.

These are orthogonal. Two markdown files can be an architecture decision record
and a changelog: identical chunking, opposite usefulness to the scenario above.
So this is a second field, not a widening of the first, and `FileKind` and the
chunker registry are untouched by any of it.

## Decisions

| Concern | Decision |
|---|---|
| Where it lives | A `doc_type` payload field, indexed, one classification per file |
| Granularity | Per file, not per chunk — every chunk of a file inherits it |
| How it is decided | A chain: rules → embedding prototypes → model, cheapest first |
| First implementation | Rules only. A model is rung three, added when the eval says rules are not enough |
| Local generative model | Skipped. If rules and prototypes are not enough, go to Haiku directly — see "On a local model" |
| Retrieval use | Boost and explicit tool choice. **Not** a silent default filter |
| Taxonomy exposure | An MCP resource *and* a tool, dynamic, with counts and real examples |
| Stability | The taxonomy becomes a public API once agents filter on it, so it is versioned |

## Taxonomy

Small on purpose. Finer categories make both rules and models unreliable, and
make it impossible to write an eval set a human can agree with.

| Type | Answers | Typical sources |
|---|---|---|
| `normative` | How it **must** be built | specs, standards, ADRs, conventions, `CONTRIBUTING` |
| `design` | How it is shaped, and why | architecture docs, RFCs, design proposals |
| `guide` | How to use or operate it | README, tutorials, runbooks |
| `reference` | What exists | API docs, generated documentation |
| `record` | What happened | changelogs, postmortems, meeting notes |
| `implementation` | The code itself | source files |
| `test` | Verification | test files, fixtures, snapshots |
| `generated` | Machine-written | build output, generated clients, lockfiles |

`test` and `generated` are pure rules and immediately let us delete the reranker
instruction hack, replacing a hope with a filter.

## The classifier chain

A `DocumentClassifier` protocol, the same shape as `Reranker`, with
implementations chained cheapest-first. Each returns a type **and a confidence**,
so the chain knows when to escalate rather than guessing.

### Rung 1 — rules. Free, deterministic, and most of the value

The signals in this domain are unusually strong:

- **Path conventions** — `docs/adr/`, `specs/`, `rfcs/`, `CONTRIBUTING.md`,
  `**/test_*`, `**/generated/`, `.claude/`
- **Frontmatter** — ADR and spec templates routinely declare `type: adr`,
  `status: accepted` outright
- **Heading skeleton** — ADRs have *Context / Decision / Consequences*; RFCs have
  *Motivation*
- **RFC-2119 modal density** — the density of `MUST`, `SHALL`, `SHOULD`,
  `MUST NOT` per 1000 words. This is the linguistic fingerprint of a
  specification, and it separates normative documents from descriptive ones
  well. A document telling you what to do reads differently from one telling you
  what exists.

### Rung 2 — embedding prototypes. Zero additional inference

We already embed every chunk. Embed a one-line description of each category
once, then compare a file's first chunk vector against those prototypes.

No new model, no new inference, no added indexing time — it is a cosine against
eight vectors we compute anyway. It will not beat an LLM, but it is close to
free and it handles files whose path and structure say nothing.

### Rung 3 — a model, on the residue only

Rules and prototypes will decide most files confidently. Send the model only
what they cannot, which should be a small minority. That inverts the cost:
hundreds of classifications rather than thousands.

## What this actually costs

**A correction to an earlier estimate.** When first discussing this I put the
model-classification cost at "200ms × 10,000 files", which conflated files with
chunks. Classification is **per file**. The measured shape of this workspace:

| Measure | Value |
|---|---|
| Files discovered | 1,003 |
| Chunks produced | 9,970 |
| Local embedding throughput | 1,863 chunks in ~35 min (~53 chunks/min) |
| Projected full index, local model | roughly 3 hours |

So classification is against **1,003 files, not 9,970 chunks** — an order of
magnitude less work than I first said. A local model at 1–3s per document is
roughly 20–50 minutes; at rung 3 on a residue of ~15% it is minutes.

This weakens the speed argument for rules-first, so the honest case rests on the
other three reasons:

1. **Determinism.** Rules give the same answer every run. A model does not, and
   see "drift" below.
2. **Auditability.** "This is `normative` because it is under `docs/adr/`" is
   checkable. "The model said so" is not.
3. **The taxonomy is the hard part, not the inference.** Deciding whether
   `docs/architecture/decisions/` is `design` or `normative`, and being
   consistent about it, is where this succeeds or fails. Rules force that
   decision to be written down.

Do not trust the projection above — measure it. `--dry-run` already reports a
plan without spending anything, and the eval harness already exists to say
whether rung 3 buys anything rungs 1 and 2 did not.

### On a local generative model

The original instinct was a small local model first, Haiku later. I would skip
the local generative step. Rung 2 already gives a local, zero-cost signal
without a generative model at all, and at rung-3 volumes a small local model is
both slower on a VM CPU and materially worse than Haiku for a task that is
mostly reading a document and picking one of eight labels. If rungs 1 and 2 are
not enough, the residue is small enough that the API cost is trivial.

## Storage and invalidation

- `doc_type` and `doc_type_confidence` on the chunk payload, `doc_type` indexed
  as a keyword so it can be filtered and counted server-side.
- Cached in the manifest **keyed by content hash**, so classification is
  incremental exactly like embedding: unchanged bytes are never reclassified.
- A `classifier_version` alongside `chunker_version`, so changing the rules
  invalidates precisely the files it should.

**Drift is the risk to design against.** If a model classifies an unchanged file
differently across runs, chunk payloads change and the store gets pointless
rewrites. Hash-keyed caching prevents it, but only if it is built in from the
start rather than added later.

## The MCP server

This is where the classification is spent, and the design deliberately avoids
inferring intent from query text.

### Tools

- **`search_code`** — implementation, with `test` and `generated` excluded by
  default. Replaces the reranker instruction hack.
- **`find_guidance`** — `normative`, `design` **and `guide`** [revised]. This is
  the greenfield case: no code to imitate, so retrieve the rules.

  `guide` was not in the plan. Normative + design alone scored recall 0.812 /
  MRR 0.792 over the eight guidance cases — *no better than plain search*
  (0.812 / 0.792). The filter gained `CLAUDE.md`, which plain search had never
  once retrieved, and lost `CONTRIBUTING.md` entirely, because a contributing
  guide is a `guide`. Adding the type gives **0.938 / 0.900**.

  The eval case that caught this was written before the tool existed, with the
  note *"deliberately a `guide`, not `normative`. If a type filter is ever
  applied by default, this case is the one that catches it over-filtering."*
  It did exactly that. Writing the dataset first is what made a plausible,
  tidy-looking design fail loudly instead of quietly.
- **`get_file_context`** — expand around a hit using `chunk_index` /
  `chunk_total`, which the payload already carries.
- **`list_document_types`** — the taxonomy, as a tool.

Separate tools rather than one tool with an intent argument, because **the agent
already knows its intent** and declaring it is far more reliable than us
inferring it from a query string.

### Measured outcome [revised]

`find_guidance` over the eight guidance cases, against plain search on the same
eight:

| | recall@10 | MRR@10 | misses |
|---|---|---|---|
| plain search | 0.812 | 0.792 | 2/8 |
| find_guidance (normative + design) | 0.812 | 0.775 | 2/8 |
| **find_guidance (+ guide)** | **0.938** | **0.900** | **1/8** |

Per case, against plain search: one previously-unfixable miss became rank 1
(*"what conventions must I follow when structuring a new module"* → `CLAUDE.md`,
which had survived every embedding and reranking configuration tried), six
unchanged at rank 1, and one regression (*"how should a new agent provider be
designed"*, rank 3 → 5).

That is the evidence #8 and #9 were gated on: rules-based classification does
carry the guidance case. The remaining miss — *"how should this project handle
logging and observability"* — is a recall problem rather than a ranking one,
and a better classifier is not obviously the fix for it.

### The taxonomy resource

A resource at `workspace-indexer://taxonomy`, **and** the same content as the
`list_document_types` tool. Both, because resources are the semantically correct
surface but clients differ in how reliably a model sees one without the user
attaching it, whereas a tool is always in context.

It must be **dynamic** — what is actually in this index, not the hardcoded enum:

```json
{
  "taxonomy_version": 1,
  "types": [
    {
      "name": "normative",
      "count": 47,
      "definition": "Specifies how things must be built: specs, standards, ADRs, conventions.",
      "examples": ["docs/adr/0007-event-sourcing.md", "specs/api-conventions.md"]
    },
    { "name": "design", "count": 12, "...": "..." }
  ]
}
```

Three things earn their place here:

- **Counts do real work.** `normative: 47` tells an agent to lean on the specs.
  `normative: 0` tells it there is no written guidance and it should fall back to
  reading existing code. **The absence is itself information**, and nothing
  communicates that today.
- **Real examples from this workspace** ground the categories better than any
  prose definition. Seeing `docs/adr/0007-event-sourcing.md` calibrates a model
  instantly.
- **`taxonomy_version`** because once agents filter on `normative`, renaming that
  category is a breaking change.

### Failure modes to design against

**Silent empty results are the worst outcome.** An agent passes `type=spec`, the
taxonomy says `normative`, it gets nothing back and concludes the workspace has
no guidance — when it merely used the wrong word.

- Put the type list in the **tool description**, so it is in context with no
  round trip at all.
- On an unknown type, **return an error naming the valid ones** — "unknown type
  'spec'; did you mean 'normative'? Available: normative, design, guide…" — never
  an empty result set.
- Accept a small **alias map** (`spec` → `normative`, `adr` → `normative`) so
  near-misses simply work.
- **Boost, do not silently filter**, outside the explicit tools. A
  misclassification that excludes the one document you needed is worse than no
  classification at all.

## Why this before the watcher

The watcher makes the index *fresher*. The classifier and MCP server make it
*usable by an agent at all*. Iteration 1 produces an index that a human can
query from a terminal; nothing yet lets Claude Code reach it during a session,
which is the entire point of the project. Freshness matters once the thing is
being used, not before.

They also pair naturally: the MCP layer is where intent is declared, which is
what makes the classification worth having.

## Build order and verification

1. **Eval cases first.** Write ~10 cases shaped like the motivating scenario —
   *"how should I structure a new service"* → expects the spec and architecture
   documents. Run them against today's index to establish the baseline.
2. **Rung 1 rules**, with the taxonomy written down. Re-run the eval. This
   number decides whether rungs 2 and 3 are worth building at all.
3. **`doc_type` into the payload and the manifest**, with hash-keyed caching and
   `classifier_version`.
4. **MCP server** with the four tools and the taxonomy resource.
5. **Rung 2 prototypes**, measured against the same eval.
6. **Rung 3**, only if the eval still says the gap is real.

Verification beyond the eval:

- Classifying an unchanged workspace twice produces byte-identical payloads —
  the drift check.
- `list_document_types` on a workspace with no specs reports `normative: 0`
  rather than omitting the category, so the absence is visible.
- An unknown type returns a helpful error, not an empty list. This is the test
  that stops the silent-empty-result failure mode from shipping.
- Deleting the reranker `instruction` hack does not regress the eval, because
  `test` and `generated` are now filtered properly.

## Open questions — resolved

**Where do `.claude/` files sit? → `normative`.** They bind how work is done in
a repository, which is what `find_guidance` should surface for "what
conventions apply here". A separate `meta` category would fragment the taxonomy
for no retrieval benefit, and the eval case *"what conventions must I follow
when structuring a new module"* expects `CLAUDE.md`, which settles it.

**Should `doc_type` be multi-valued? → the question dissolves.** Qdrant keyword
payload fields accept arrays natively and `MatchValue` matches membership —
proven by our own `ancestors` field, which is already a list under a `KEYWORD`
index. So storing a single string now and a list later is a change in what we
write, not a schema migration. Single-valued to start, with no trap.

**Should the type appear in the embedded text? → still open, deliberately.**
Testable with the eval harness once results are persisted (#1). Note it changes
what is embedded but not chunk identity, since the header is excluded from
`content_sha` — so trying it costs a re-embed and nothing else.

## Remaining open questions

- Where do `.claude/` files sit? They are instructions to an agent, which is
  arguably `normative`, but they are about *how to work* rather than *what to
  build*. Possibly a `meta` category; possibly `normative` is right.
- Should `doc_type` be multi-valued? A document can be both a design record and
  a convention. Single-valued is simpler and probably correct to start, but the
  payload field should not make multi-valued impossible later.
- Does the classification belong in the embedded text as well as the payload —
  would prefixing `# type: normative` to the context header improve dense
  retrieval, or just add noise? Testable with the eval harness.
