@echo OFF
set AIDER_CHAT_LANGUAGE=English
set AIDER_COMMIT_LANGUAGE=English
set OPENAI_API_BASE=http://localhost:8080/v1
set OPENAI_API_KEY=sk-bf-4b394759-cc70-4e47-b69c-d214ba9267b2
aider --architect --model openai/tier-plan --editor-model openai/tier-build --weak-model openai/tier-triage --map-tokens 1024 --no-show-model-warnings