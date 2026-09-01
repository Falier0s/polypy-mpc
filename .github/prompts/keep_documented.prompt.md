---
name: keep documented
description: Improve and maintain code documentation by editing only comments and docstrings; record each change in a tracking file.
argument-hint: "Describe which file/area needs documentation improvements (optional)."
---

# Keep Documented (Prompt Instructions)

This prompt is meant to be used when you want to **improve or maintain documentation** inside the codebase by editing **only comments and docstrings**. The goal is to keep the implementation untouched while making the code easier to understand for both humans and tools.

---

## ✅ What you should do

1. **Find code areas needing better documentation** (comments or docstrings), especially:
   - complex logic sections
   - function/method/class interfaces
   - non-obvious assumptions or units
   - TODOs or FIXMEs that affect readability

2. **Edit only comments and docstrings**:
   - Do **not** change executable code, variable names, logic, imports, or structure.
   - Prefer clarifying *why* something is done, not just *what* is done.
   - Use consistent style (e.g., NumPy-style docstrings for functions/classes).

3. **Keep the rest of the code unchanged**. If a code change is needed, stop and ask for clarification instead of applying it.

---

## 📌 Tracking changes (required)

Every time you update comments/docstrings, you must also append a short entry to a tracking file so we can later understand what was modified and avoid re-editing the same parts again.

### Tracking file location
Create or update the file:

- `doc/documentation_updates.md`

### What to include in the tracking entry
For each session of edits, add a small report entry that includes:

- **Date (UTC)**
- **Files modified** (paths)
- **Brief summary** of what was updated (e.g., "Clarified `mpc.solve` docstring and added explanation for `r` handling").
- **Optional**: note if any of the changes look like they were later modified by a human (to ease future diff reviews).

Example entry:

```
## 2026-03-10
- Updated `mpc.py`:
  - Expanded class docstring to explain regulation vs tracking.
  - Added docstrings for all placeholder methods.
  - Clarified the meaning of `sigma` in the reachability set.
```

---

## 🛑 Important constraints (strictly enforced)

- **Only comments/docstrings may be edited.** No code changes, no refactors, no formatting changes to executable lines.
- If you believe a code fix is needed (e.g., a missing variable like `self.D` or a syntax issue), stop and request guidance instead of changing code.
- Keep the documentation clear, consistent, and written in English.

---

## 🧠 How to use this prompt

Whenever you want to improve documentation:

1. Open this prompt (`.github/prompts/keep_documented.md`).
2. Use it as the instruction set for your documentation edits.
3. Apply changes only in comments/docstrings.
4. Update `doc/documentation_updates.md` with the summary of changes.

---

✅ This ensures the project stays well documented while keeping all functional code stable and auditable.
