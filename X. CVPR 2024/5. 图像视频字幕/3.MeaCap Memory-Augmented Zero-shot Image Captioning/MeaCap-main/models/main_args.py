import argparse


def get_args():

    parser = argparse.ArgumentParser(description="Text infilling.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--bart', type=str, default='large', choices=['base', 'large'])
    parser.add_argument('--refinement_steps', type=int, default=10, help='The number of refinements for each input.')
    parser.add_argument('--adaptive', type=bool, default=False, help='The number of refinements is on the fly but '
                                                                     'no bigger than max_refinement_steps')
    parser.add_argument('--max_refinement_steps', type=int, default=30, help='The maximum number of refinements for each input.')
    parser.add_argument('--max_len', type=int, default=20, help='The maximum length of the generated sentence.')
    parser.add_argument('--min_len', type=int, default=10, help='The minimum length of the generated sentence.')
    parser.add_argument('--temperature', type=float, default=1,
                        help='The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.')
    parser.add_argument('--repetition_penalty', type=float, default=2,
                        help='Between 1.0 and infinity.1.0 means no penalty.Default to 1.0.')
    parser.add_argument('--shuffle_sample_model', type=str, default='openai/clip-vit-base-patch32',
                        help='Between 0 and 1. 0 means no threshold for copy action. Default to 0.')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Between 0 and 1. 0 means no threshold for copy action. Default to 0.')

    parser.add_argument('--top_k', type=int, default=0,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity.')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. '
                             'Must be between 0 and 1.')
    parser.add_argument('--decoder_chain', type=int, default=1,
                        help='the number of parallel chains for decoder, each chain refers to an unique token sequence.')
    parser.add_argument('--do_sample', type=int, default=0,
                        help='if 0 decode with greedy method, otherwise decode with top_k or top_p.')
    parser.add_argument('--encoder_loss_type', type=int, default=0, help='0 is classification loss, 1 is regression loss')
    parser.add_argument('--dataset', type=str, default='COCO',
                        choices=['yelp_review', 'one-billion-words'])
    parser.add_argument('--insert_mode', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='0 means using the left part, 1 means using the middle part, 2 means using the right part,'
                             '3 means randomly selecting, 4 means selecting the tokens with highest weight')
    parser.add_argument('--max_insert_label', type=int, default=1, help='the maximum number of tokens to be inserted before a token.')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='0 for copy, 1 for replace, 2-5 means insert 1-4 tokens')
    parser.add_argument('--generate_mode', type=int, default=0, choices=[0, 1, 2, 3],
                        help='0 for random, 1 for lm, 2 for combination')
    parser.add_argument('--masked_lm', type=float, default=0, help='0 for using language modeling for the decoder,'
                                                                   '1 for using mask language modeling for the decoder.')
    parser.add_argument('--full_mask', type=float, default=0, help='0 for using casual mask attention for decoder, '
                                                                   '1 for without using casual mask attention for decoder.')
    parser.add_argument('--w', type=float, default=1.0, help='The weight for the encoder loss')

    parser.add_argument('--num_keywords', type=int, default=4, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--random_init', type=int, default=0, help='0 denotes initialization with BART; '
                                                                   '1 denotes random initialization.')

    parser.add_argument('--conzic_sample', type=bool, default=True, help='conzic sample means a way to process logits by conzic method')
    parser.add_argument('--clip_model', type=str, default=r'openai/clip-vit-base-patch32')
    parser.add_argument('--img_path', type=str, default=r'path/to/images')
    parser.add_argument('--conzic_top_k', type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1, help="weight for fluency")
    parser.add_argument("--beta", type=float, default=0.4, help="weight for image-matching degree")
    parser.add_argument("--gamma", type=float, default=0.2, help="weight for fluency")
    parser.add_argument("--memory_caption_num", type=int, default=5)
    parser.add_argument("--shuffle_alpha", type=float, default=0.1, help="shuffle sample weight for memory similarity")
    parser.add_argument("--shuffle_beta", type=float, default=0.9, help="shuffle sample weight for image similarity")

    parser.add_argument("--use_detect", type=bool, default=True)
    parser.add_argument("--wordnet", type=bool, default=False, help="use world Net to expand keywords")
    parser.add_argument("--save_path", type=str, default='./demo_results')
    parser.add_argument("--use_prompt", type=bool, default=True)
    parser.add_argument("--prompt", type=list, default=['An image of'])
    parser.add_argument("--use_memory", type=bool, default=True)
    parser.add_argument("--memory_caption_file", type=str, default='../data/memory/cc3m/train_captions.json')
    parser.add_argument("--memory_clip_embedding_file", type=str, default='../data/memory/cc3m/train_clip_embedding.pt')
    parser.add_argument("--memory_wte_embedding_file", type=str, default='../data/memory/cc3m/train_wte_embedding.pt')
    parser.add_argument("--parser_checkpoint", type=str, default=r'lizhuang144/flan-t5-base-VG-factual-sg')
    parser.add_argument("--wte_model_path", type=str, default=r'sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument("--model_path", type=str, default=r'path/to/model')
    args = parser.parse_args()

    return args
