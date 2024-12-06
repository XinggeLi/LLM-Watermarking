import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from watermark_processor_v2 import WatermarkLogitsProcessor, WatermarkDetector
from functools import partial
import math
from tqdm import tqdm

def load_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

def add_idx(example, idx):
    example.update({"idx":idx})
    return example

def parse_tokenize_prompts(args, model, tokenizer):
    tokenize_prompts_kwargs = {
        'tokenizer': tokenizer,
        'model_max_seq_len': model.config.max_position_embeddings
    }
    if args.input_truncation_strategy == "prompt_length":
        tokenize_prompts_kwargs['min_prompt_tokens'] = args.min_prompt_tokens
    elif args.input_truncation_strategy == "completion_length":
        tokenize_prompts_kwargs['max_new_tokens'] =args.max_new_tokens
    else:
        ValueError(f"Unknown input truncation strategy {args.input_truncation_strategy}")
    return partial(tokenize_prompts, **tokenize_prompts_kwargs)

def tokenize_prompts(
    example, 
    idx,
    max_new_tokens = None,
    min_prompt_tokens = None,
    tokenizer = None,
    model_max_seq_len = 4096,
):
    untruncated_input_encoded = tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)

    if (max_new_tokens is not None) and (min_prompt_tokens is None):
        slice_length = min(untruncated_input_encoded.shape[1] - 1, max_new_tokens)
    elif (min_prompt_tokens is not None) and (max_new_tokens is None):
        desired_comp_len = (untruncated_input_encoded.shape[1] - 1) - min_prompt_tokens
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError((f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                          f" but got completion_length:{max_new_tokens}, prompt_length:{min_prompt_tokens}"))
    
    input_encoded = untruncated_input_encoded[:,:untruncated_input_encoded.shape[1] - slice_length]

    input_decoded = tokenizer.batch_decode(input_encoded, skip_special_tokens=True)[0]

    untruncated_input_decoded = tokenizer.batch_decode(untruncated_input_encoded, skip_special_tokens=True)[0]

    example.update({
        "untruncated_input_encoded"         : untruncated_input_encoded,
        "untruncated_input_encoded_length"  : untruncated_input_encoded.shape[1],
        "input_encoded"                     : input_encoded,
        "input_encoded_length"              : input_encoded.shape[1],
        "input_decoded"                     : input_decoded,
        "real_completion_decoded"           : untruncated_input_decoded[len(input_decoded):],
        "real_completion_encoded_length"    : untruncated_input_encoded.shape[1] - input_encoded.shape[1],
    })
    return example

def parse_input_check(args):
    input_check_kwargs = {'min_untruncated_input_encoded_length': args.min_untruncated_input_encoded_length}
    if args.input_filtering_strategy == "prompt_length":
        input_check_kwargs.update({
            'min_input_encoded_length': args.min_input_encoded_length,
            'min_real_completion_encoded_length': 0
        })
    elif args.input_filtering_strategy == "completion_length":
        input_check_kwargs.update({
            'min_input_encoded_length': 0,
            'min_real_completion_encoded_length': args.max_new_tokens
        })
    elif args.input_filtering_strategy == "prompt_and_completion_length":
        input_check_kwargs.update({
            'min_input_encoded_length': args.min_input_encoded_length,
            'min_real_completion_encoded_length': args.max_new_tokens
        })
    else:
        ValueError(f"Unknown input filtering strategy {args.input_filtering_strategy}")
    return partial(input_check, **input_check_kwargs)

def input_check(
    example, 
    idx, 
    min_untruncated_input_encoded_length=0, 
    min_input_encoded_length=0, 
    min_real_completion_encoded_length=0
):
    return all([
        example["untruncated_input_encoded_length"] >= min_untruncated_input_encoded_length,
        example["input_encoded_length"] >= min_input_encoded_length,
        example["real_completion_encoded_length"] >= min_real_completion_encoded_length,
    ])

def parse_output_check(args):
    if args.output_filtering_strategy == "max_new_tokens":
        output_check_kwargs = {'min_output_len': args.max_new_tokens}
    elif args.output_filtering_strategy == "no_filter":
        output_check_kwargs = {'min_output_len': 0}
    else:
        ValueError(f"Unknown output filtering strategy {args.output_filtering_strategy}")
    return partial(output_check, **output_check_kwargs)

def output_check(example,min_output_len=0):
    return all([
        example["completion_wo_wm_encoded_length"] >= min_output_len,
        example["completion_w_wm_encoded_length"] >= min_output_len,
    ])

def parse_gen_completions(args, model, tokenizer):
    wm_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme
    )

    logit_processor_list = LogitsProcessorList([wm_processor])

    gen_kwargs = {
        'max_new_tokens': args.max_new_tokens, 
        'num_beams': args.n_beams,
        'pad_token_id': tokenizer.eos_token_id
    }
    if args.n_beams > 1:
        if args.no_repeat_ngram_size > 0:
            gen_kwargs['no_repeat_ngram_size'] = args.no_repeat_ngram_size
        gen_kwargs['early_stopping'] = args.early_stopping

    if args.use_sampling:
        gen_kwargs.update({
            'do_sample': True,
            'top_k': 0,
            'temperature': args.sampling_temp})

    gen_wo_wm = partial(
            model.generate,
            **gen_kwargs
        )
    
    gen_w_wm = partial(
        model.generate,
        logits_processor=logit_processor_list, 
        **gen_kwargs
    )

    return partial(
        gen_completions,
        tokenizer=tokenizer,
        model=model,
        wo_wm_partial=gen_wo_wm,
        w_wm_partial=gen_w_wm,
    )

def gen_completions(
    example,
    idx,
    tokenizer=None,
    model=None,
    w_wm_partial=None,
    wo_wm_partial=None,
):
    input_encoded = example["input_encoded"]
    input_decoded = example["input_decoded"]
    with torch.no_grad():
        samples_taken = 0
        max_retries = 10
        success = False
        while (success is False) and (samples_taken < max_retries):
            samples_taken += 1
            output_wo_wm_encoded = wo_wm_partial(input_encoded.to(model.device))
            output_w_wm_encoded = w_wm_partial(input_encoded.to(model.device))

            try:
                output_wo_wm_decoded = tokenizer.batch_decode(output_wo_wm_encoded, skip_special_tokens=True)[0]
                example["completion_wo_wm_decoded"] = output_wo_wm_decoded[len(input_decoded):]

                output_w_wm_decoded = tokenizer.batch_decode(output_w_wm_encoded, skip_special_tokens=True)[0]
                example["completion_w_wm_decoded"] = output_w_wm_decoded[len(input_decoded):]
                
                success = True
            except:
                # log what happened
                print(f"Error while trying to decode the outputs of the model...")
                if samples_taken == 1:
                    print(f"truncated_input: {input_encoded.tolist()}")
                print(f"Result of attempt {samples_taken}")
                print(f"shape output_wo_wm_encoded: {output_wo_wm_encoded.shape}")
                print(f"shape output_w_wm_encoded: {output_w_wm_encoded.shape}")

        if success is False:
            print(f"Unable to get both a wo_wm and w_wm output that were decodeable after {samples_taken} tries, returning empty strings.")
            example["completion_wo_wm_decoded"] = ""
            example["completion_w_wm_decoded"] = ""

    example.update({
        "completion_wo_wm_encoded_length"    : output_wo_wm_encoded.shape[1] - input_encoded.shape[1],
        "completion_w_wm_encoded_length"     : output_w_wm_encoded.shape[1] - input_encoded.shape[1]
    })
    
    return example

def parse_result_ppl_eval(oracle_model, oracle_tokenizer):
    return partial(
        result_ppl_eval,
        oracle_model=oracle_model,
        oracle_tokenizer=oracle_tokenizer
    )

def result_ppl_eval(
    example, 
    idx,
    oracle_model = None,
    oracle_tokenizer = None
):
    input_plus_real_completion_decoded = f"{example['input_decoded']}{example['real_completion_decoded']}"
    real_completion_decoded = f"{example['real_completion_decoded']}"

    input_plus_completion_wo_wm_decoded = f"{example['input_decoded']}{example['completion_wo_wm_decoded']}"
    completion_wo_wm_decoded = f"{example['completion_wo_wm_decoded']}"

    input_plus_completion_w_wm_decoded = f"{example['input_decoded']}{example['completion_w_wm_decoded']}"
    completion_w_wm_decoded = f"{example['completion_w_wm_decoded']}"

    
    real_loss, real_ppl = ppl_eval(input_plus_real_completion_decoded, real_completion_decoded, oracle_model, oracle_tokenizer)
    wo_wm_loss, wo_wm_ppl = ppl_eval(input_plus_completion_wo_wm_decoded, completion_wo_wm_decoded, oracle_model, oracle_tokenizer)
    w_wm_loss, w_wm_ppl = ppl_eval(input_plus_completion_w_wm_decoded, completion_w_wm_decoded, oracle_model, oracle_tokenizer)
    
    example.update({
        'real_loss' : real_loss,
        'real_ppl'  : real_ppl,
        'wo_wm_loss': wo_wm_loss,
        'wo_wm_ppl' : wo_wm_ppl,
        'w_wm_loss' : w_wm_loss,
        'w_wm_ppl'  : w_wm_ppl
    })
    
    return example

def ppl_eval(
    input_plus_completion_decoded = None,
    completion_decoded = None,
    oracle_model = None,
    oracle_tokenizer = None
):
    with torch.no_grad():
        tokd_inputs = oracle_tokenizer.encode(input_plus_completion_decoded, return_tensors="pt", truncation=True, max_length=oracle_model.config.max_position_embeddings)
        tokd_suffix = oracle_tokenizer.encode(completion_decoded, return_tensors="pt", truncation=True, max_length=oracle_model.config.max_position_embeddings)
        
        tokd_inputs = tokd_inputs.to(oracle_model.device)
        tokd_labels = tokd_inputs.clone().detach()

        tokd_labels[:,:tokd_labels.shape[1]-tokd_suffix.shape[1]+1] = -100

        outputs = oracle_model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss
        ppl = torch.tensor(math.exp(loss))
    
    return loss.item(), ppl.item()

def perform_detection(args, tokenizer, device, result_df):
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        z_threshold=args.z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams
    )
    for i, row in tqdm(result_df.iterrows()):
        text = str(row.get('completion_decoded', ''))

        result_df.at[i, 'prediction'] = None
        result_df.at[i, 'confidence'] = None
        result_df.at[i, 'p_value'] = None
        result_df.at[i, 'z_score'] = None
        result_df.at[i, 'green_fraction'] = None
        result_df.at[i, 'num_green_tokens'] = None
        result_df.at[i, 'num_tokens_scored'] = None
        
        if not text.strip():
            print(f"Empty/Missing row {i}")
            continue

        try:
            tokenized_text = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)
        except Exception as e:
            print(f"Error tokenizing row {i}: {e}")
            continue

        d = {}
        if len(tokenized_text) > 0:
            if tokenized_text[0] == tokenizer.bos_token_id: # remove Beginning of Sequence (BOS)
                tokenized_text = tokenized_text[1:]
            if len(tokenized_text) - 1 > watermark_detector.min_prefix_len:
                d = watermark_detector.detect(tokenized_text)
        result_df.at[i, 'prediction'] = d.get('prediction')
        result_df.at[i, 'confidence'] = d.get('confidence')
        result_df.at[i, 'p_value'] = d.get('p_value')
        result_df.at[i, 'z_score'] = d.get('z_score')
        result_df.at[i, 'green_fraction'] = d.get('green_fraction')
        result_df.at[i, 'num_green_tokens'] = d.get('num_green_tokens')
        result_df.at[i, 'num_tokens_scored'] = d.get('num_tokens_scored')
    return result_df