# SAARTHI prototype (software core)

Storage, tools, router, cause ranking, intrusion detector, and the model path with the base 0.5B model.
Runs on any machine with Python 3.10+.

    pip install duckdb numpy pandas pyarrow scikit-learn
    python3 gen_history.py                 # 6 weeks of synthetic vehicle data, three storage tiers
    taskset -c 0,1 python3 saarthi.py      # answers the five questions from the brief
    taskset -c 0,1 python3 evaluate.py     # router accuracy, latency, cause ranking, memory -> results.json
    taskset -c 0,1 python3 can_ids.py      # intrusion detector on synthetic CAN traffic -> ids_results.json

Model path (needs: pip install llama-cpp-python, and the GGUF file from
huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF, file qwen2.5-0.5b-instruct-q4_k_m.gguf):

    taskset -c 0,1 python3 model_path.py qwen2.5-0.5b-instruct-q4_k_m.gguf   # -> model_results.json

Ask your own question: `python3 saarthi.py "tyre pressure ka trend batao"`

Files
- gen_history.py  vehicle energy model, fault injection, Parquet + DuckDB tiers
- saarthi.py      fast-path router, validator, seven tools, answer templates
- evaluate.py     69 hand-written test questions and 10 fault scenarios
- can_ids.py      CAN intrusion detector (DoS, fuzzing, spoofing)
- model_path.py   base Qwen2.5-0.5B with a grammar-constrained tool call, 2 threads

Results in this folder were measured on an x86 development machine pinned to 2 cores.
To do on the Pi: repeat all of these runs there. The same commands work on a Raspberry Pi 4
(64-bit OS). Then fine-tune the model and swap the GGUF file.
