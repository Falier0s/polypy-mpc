---
name: auto readme
description: Generate or update a README based on the documentation changelog and the current Python modules in the repository.
argument-hint: "Optionally specify a target README filename or section to update."
---

# Auto README Generator (Prompt Instructions)

This prompt is designed to automatically generate or update a **README** file that describes the current Python modules and the state of documentation in the repository.

It should use two main sources:

1. **`doc/documentation_updates.md`** — the changelog tracking what documentation has been edited.
2. **All Python modules** found in the repository — to summarize available functionality.

---

## ✅ What this prompt should do

1. **Read the documentation log** (`doc/documentation_updates.md`) and extract a high-level summary of what has been documented and when.
2. **Inspect Python modules** in the repo (all `.py` files) and identify:
   - Main classes and functions
   - Purpose of each module (from top-level docstrings or patterns)
   - Any notable usage examples or entry points
3. **Generate an `README.md`** (or update an existing one) that includes:
   - A short project description
   - A summary of documented changes (from the log)
   - A module/function overview with brief explanations
   - Basic usage instructions or example snippets (if inferable)

---

## 📝 Output requirements

- Produce a complete Markdown file suitable as a repository README.
- Use English only.
- Focus on clarity and navigability for a new reader.
- Do not modify any source code; only produce/edit the README content.

---

## 🛑 Important constraints

- **Do not change any Python code.** This prompt is only for generating documentation.
- If a module lacks docstrings or is ambiguous, make a reasonable guess and note it clearly.
- If the repository already has a `README.md`, update it; otherwise, create a new one.

---

## 📌 How to use

Run this prompt whenever you want to regenerate a README that reflects the latest documented changes and the current module set.

If you want the prompt to target a specific README file (e.g. `README.md` vs `doc/README.md`), specify that in the input.
