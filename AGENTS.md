# Python conventions

- Check Python code with Pyright in strict mode. Resolve type errors before considering a change complete.
- Declare explicit types for variables, including local variables and attributes. Annotate function parameters and return values. For bindings that Python cannot annotate inline, such as loop targets, declare the variable's type before the binding.
- Spell out English words in identifiers. Use descriptive names such as `index`, `context`, and `count`. Avoid abbreviations such as `idx`, `ctx`, and `cnt`.
- Write comments, docstrings, prompts, error messages, and other program text in English. Follow the wording and tone of comparable Python standard library messages. Describe errors precisely and name the offending argument or value when helpful.

## KayKit integration

- Integrate content from `kaykit.7z` into the game's existing Model-View-Presenter (MVP) architecture. Keep gameplay state in the model, coordinate actions in the presenter, and render the character in the view.
- Reuse the model loading and rendering code provided in the archive's `kaykit/kaykit_previewer/` package. Use `application.py` as the reference for advancing animation playback, `animation.py` for evaluating poses and computing skinning matrices, and `renderer.py` for drawing the animated model.
