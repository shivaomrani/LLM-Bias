# LLM-Bias
Code implementation of the paper "Evaluating Biased Attitude Associations of Language Models in an Intersectional Context" published in the 6th AAAI/ACM Conference on AI, Ethics, and Society (AIES 23). If you have any questions about the code, please contact `somrani@gwu.edu`.
 
## Running Instructions
To run the code, navigate to `src` directory and run `python main.py <experiment name> <name of LLM if applicable> `
- The possible values for `<experiment name>` are:
  - `experiment1`: generates results for experiment 1 (Evaluating Learned Affective Dimensions Against Human Judgments of Semantics) of the paper
  - `experiment2`: generates results for experiment 2 (Bias Evaluation Using SC-WEAT) of the paper. It also produces LaTeX code for generating figure 3 in the paper for different models. 
  - `experiment3`: generates results for experiment 3 (Identifying the Strongest Biases Across Contexts) of the paper
  - `generate_table`: generates the LaTeX code for producing table #2 from the paper which summarizes the results from experiment 2 for all five language models. 
  Note that in order for this to run, you need to run experiment 2 for all language models individually.
- The possible values for `<name of LLM>` are the five LLMs studied in our paper, which are:
    - `albert`
    - `gptneo`
    - `roberta`
    - `t5`
    - `xlnet`
    - you can leave this field blank for `generatee_table` experiment

For instance, to run experiment 1 for gptneo, you run `python experiment1 gptneo`.
