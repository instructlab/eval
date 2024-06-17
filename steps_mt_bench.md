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

You should run with `--tensor-parallel-size <NUM GPUS>` and possibly increase `--max-model-len` to increase the context length

```shell
python -m vllm.entrypoints.openai.api_server --model instructlab/granite-7b-lab --served-model-name gpt-4
```

```shell
OPENAI_API_BASE=http://0.0.0.0:8000/v1 OPENAI_API_KEY="NO_API_KEY" python src/instructlab/eval/gen_judgment.py --bench-name mt_bench --parallel 40 --yes
```

results are in data/mt_bench/model_judgment/gpt-4_single.jsonl

After this is over we run the following to get the full score:

```shell
python src/instructlab/eval/show_result.py --bench-name mt_bench
```

output looks like

```shell
Mode: single
Input file: data/mt_bench/model_judgment/gpt-4_single.jsonl

########## First turn ##########
                        score
model          turn          
granite-7b-lab 1     7.621622

########## Second turn ##########
                        score
model          turn          
granite-7b-lab 2     4.666667

########## Average ##########
                score
model                
granite-7b-lab    7.4
```
