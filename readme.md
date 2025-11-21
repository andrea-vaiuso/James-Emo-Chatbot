## James – Emotional Chatbot

A modular, emotion-aware chatbot with persistent memory, persona selection, and a Tkinter chat UI.

### Quick Start
- Run `python main.py` to pick a personality and start chatting.
- Exit a chat with the `(Back)` button in the header; conversation saving happens asynchronously.
- Personalities live in `Personalities/`; user profile in `user_profile.json`; session memory in `Memory/`.

### Architecture at a Glance
- `main.py`: app bootstrap, personality selection, async conversation save/return flow.
- `bubbleChat.py`: Tkinter chat UI, typing bubble, message batching, periodic emotion refresh.
- `LLM/llm.py`: model wrapper, system prompt assembly, response generation.
- `LLM/memory.py`: memory persistence, short/long-term memories, timing, emotional state updates.
- `LLM/emotions.py`: emotion analysis pipeline (`analyze_emotional_state`).

### Memory System
- Stored per persona in `Memory/{ai_username}_memory.json`.
- **Meta**: `first_interaction`, `last_interaction` (UTC), `last_update`, `user_id`.
- **User profile**: name/age/gender enrichment.
- **Short term**: rolling `memory_chunks` (recent messages) summarized into `short_term_memory.memories`.
- **Long term**: facts in `long_term_memory.memories`; persistent emotions in `persistent_emotions`.
- **Current affect**: `current_affect_state` with `value` ∈ [0,1] and `reasons` list per emotion.
- Update flow (pseudo-schema):
  - On message batches: `update_current_emotional_state(text)` → smooth values, append capped reasons.
  - On conversation end: short-term summarization → possible long-term fact → `last_interaction` stamped.

#### Value Smoothing
Exponential moving average per emotion:
```
v_new = α * v_instant + (1 - α) * v_old
```
used in `update_emotional_value`, default α=0.6.

#### Reason Retention
`update_current_emotional_state(text, max_reasons=3)`:
- Extends existing reasons with newly detected ones.
- Keeps the latest `max_reasons` entries per emotion (no overwrite of older until cap exceeds).

### Emotional Model Flow
1) `analyze_emotional_state` extracts `{emotion: {"value": float, "reasons": [..]}}`.
2) `update_current_emotional_state` smooths values and appends reasons.
3) `generate_response(..., include_emotional_context=True)` can prepend a system emotional context from the top emotions.
4) Emotions also influence memory summaries stored with short/long-term facts.

### Timing & Last Interaction
- UTC timestamps via `memory.now()`.
- `get_timing_information_prompt` builds a human-readable delta (“Last interaction X days, Y hours ago”); if over a week it notes it’s been a while.
- Header in chat shows the last interaction line under the bot name.

### UI Notes
- `(Back)` button cancels in-flight generations, triggers async save, and returns to personality selection.
- Typing bubble hides when you type a follow-up and reappears when the bot is generating.
- If multiple user messages arrive before a bot reply, they are concatenated and answered together.

### Extending
- Add new emotions: adjust `LLM/emotions.py` classifiers and ensure `current_affect_state` defaults exist in the memory template.
- Tune responsiveness: adjust `N` in `bubbleChat.py` for the message window used in emotion updates.
- Swap model: update `model_name` in `LLM/llm.py` and ensure tokenizer/model are available.


