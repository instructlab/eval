## 0.4.2

* Adds the ability to provide a custom system prompt to the MMLU-based evaluators. When a system prompt is provided, LM-eval applies the chat template under the hood, else it will pass the model a barebones prompt.
* Adds an `extra_args` parameter to the `.run` method of all MMLU-based evaluators. This way, consumers are able to directly pass any additional arguments they want through to the `lm_eval.evaluators.simple_evaluate` function.

## 0.4

* Added ability to specify a custom http client to MT-Bench

## v0.2
