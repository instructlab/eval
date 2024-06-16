# MT-Bench Broken Down in Eval Scripts (not PR Bench)

## prepare steps

### From justfile `./just prepare_bench ws-mt`

Calls into `./scripts/prepare_fschat_bench.sh`

No args for just MT Bench

#### `prepare_fschat_bench.sh`
```shell
cd $WORKSPACE
python3.9 -m venv venv
source venv/bin/activate
pip install --quiet -U setuptools
git clone --quiet https://github.com/shivchander/FastChat.git
cd $WORKSPACE/FastChat
git switch ibm-pr # TODO
pip install --quiet --use-pep517 .
pip install --quiet pandas torch transformers accelerate openai==0.28.0 anthropic
sed -i 's/NEED_REF_CATS = \[/NEED_REF_CATS = \["taxonomy", /g' $WORKSPACE/FastChat/fastchat/llm_judge/common.py
sed -i 's/args = parser.parse_args()/parser.add_argument("--yes", action="store_true")\n    args = parser.parse_args()/g' $WORKSPACE/FastChat/fastchat/llm_judge/gen_judgment.py
sed -i 's/input("Press Enter to confirm...")/if not args.yes:\n        input("Press Enter to confirm...")/g' $WORKSPACE/FastChat/fastchat/llm_judge/gen_judgment.py
```

### From justfile `./just create_venv`

```shell
pip install --quiet wandb ibm-watson matplotlib tqdm pandas pygithub ibmcloudant tenacity
```

## running steps

### `./just link_rc $ORG_NAME $MODEL_NAME-rc $RC_MODEL_PATH`

Not relevant to just MT-Bench

## start local step 

### From justfile `./just start_local $MODEL_NAME $ORG_NAME`

Not relevant to library because we will assume a running model as an input OR start it ourselves?

## run bench judge step 

## From justfile `./just run_bench_judge ws-mt $MODEL_NAME $DRY_RUN mt_bench`

Breaks down into `run_bench` and `run_judge` in the `justfile`

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

### From justfile: `run_judge`

