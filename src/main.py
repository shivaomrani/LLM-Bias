import sys
import warnings
warnings.filterwarnings("ignore")

def main():
    transformers_ = ['albert','gptneo','roberta','t5','xlnet']
    if len(sys.argv) == 3 and sys.argv[2].lower() not in transformers_:
        sys.exit("Correct usage: python main.py <experiment name> <name of LLM>")
    elif len(sys.argv) > 3 or len(sys.argv) == 1:
        sys.exit("Correct usage: python main.py <experiment name> <name of LLM>")
    elif len(sys.argv) == 2 and sys.argv[1] != "generate_table":
        sys.exit("Correct usage: python main.py <experiment name> <name of LLM>")

    if sys.argv[1] == "experiment1":
        import experiment1_valnorm
    elif sys.argv[1] == "experiment2":
        import embedding_collector
        import valence_association_measurements
        import experiment2_generate_image
    elif sys.argv[1] == "experiment3":
        import embedding_collector
        import valence_association_measurements
        import experiment3
    elif sys.argv[1] == "generate_table":
        import experiment2_generate_table
    else:
        sys.exit("invalid value for experiment number")


if __name__ == "__main__":
    main()