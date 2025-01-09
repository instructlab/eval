## 0.5.0

* Introduces Ragas as a supported evaluation framework. This integration only supports the `RubricsScore` metric and OpenAI models. Users can pass in either a dataset with a pre-computed `user_input`, `reference` and `response` fields or they can provide a dataset containing `user_input` and `reference` along with information about a model endpoint that will be used for computing the `response` field.

## 0.4.2

* Adds the ability to provide a custom system prompt to the MMLU-based evaluators. When a system prompt is provided, LM-eval applies the chat template under the hood, else it will pass the model a barebones prompt.
* Adds an `extra_args` parameter to the `.run` method of all MMLU-based evaluators. This way, consumers are able to directly pass any additional arguments they want through to the `lm_eval.evaluators.simple_evaluate` function.

## 0.4

* Added ability to specify a custom http client to MT-Bench

## v0.2
