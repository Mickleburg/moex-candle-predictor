# LLM Source

`llm_ta/` содержит runtime-код блока LLM Technical Analysis:

- `providers.py` - base provider, mock provider и OpenAI-compatible local provider.
- `analyzer.py` - prompt build, strict parse/validation и fallback.
- `validator.py` - JSON Schema validation плюс проверка суммы вероятностей.
- `cli.py` - простой CLI для smoke/integration checks.
