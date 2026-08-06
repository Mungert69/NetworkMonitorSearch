# LLM-Wiki maintenance schema

This directory is the maintained source for the repository's GitHub wiki. It is generated and revised from tracked repository evidence; application source is authoritative when it conflicts with a page.

## Conventions

- Keep one concept per Markdown page, using GitHub wiki links such as `[Operations](Operations)`.
- Update `index.md` whenever adding, renaming, or materially changing a page.
- Append, never rewrite, `log.md` for a meaningful ingest, correction, or wiki health check. Use `## [YYYY-MM-DD] type | title` headings.
- State uncertainty and point to the source file rather than inventing contracts, configuration values, or operational claims.
- Never copy credentials, production data, or raw conversation history into this directory.
- Prefer concise operational descriptions over file-by-file paraphrases. Do not document generated code unless it affects maintenance.

## Update workflow

1. Read `index.md`, the relevant current pages, and source files (including sibling shared-project contracts when necessary).
2. Change the affected pages and cross-links as one coherent edit.
3. Verify links and page names; `Home.md` is the GitHub-wiki landing page and `_Sidebar.md` is navigation.
4. Add an append-only log entry describing evidence and any remaining gap.
5. Perform a lightweight lint: find broken links, orphan pages, stale source references, untracked operational assumptions, and pages that need cross-links.

## Source boundaries

Raw source is the repository code, project files, tests, scripts, and relevant sibling project contracts. Do not modify raw code while carrying out a documentation-only ingest unless the user explicitly asks for a code change.
