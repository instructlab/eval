# MT-Bench Broken Down in Eval Scripts (not PR Bench)

### From justfile: `run_bench`

If dry run:
```shell
OPENAI_API_KEY="NO_API_KEY" python gen_api_answer.py \
--bench-name mt_bench \
--openai-api-base http://localhost:8000/v1 \
--model granite-7b-lab \
--num-choices 1 \
--question-begin 2 \
--question-end 4
```

If not dry run
```shell
OPENAI_API_KEY="NO_API_KEY" python gen_api_answer.py \
--bench-name mt_bench \
--openai-api-base http://localhost:8000/v1 \
--model granite-7b-lab \
--num-choices 1
```

results are in data/mt_bench/model_answer/instructlab/granite-7b-lab.jsonl

### From justfile: `run_judge`

For running judge model with vllm make sure you run with `--served-model-name gpt-4`

```shell
python -m vllm.entrypoints.openai.api_server --model instructlab/granite-7b-lab --served-model-name gpt-4
```

```shell
OPENAI_API_BASE=http://0.0.0.0:8000/v1 OPENAI_API_KEY="NO_API_KEY" python src/instructlab/eval/gen_judgment.py --bench-name mt_bench --parallel 40 --yes
```

results are in data/mt_bench/model_judgment/gpt-4_single.jsonl
