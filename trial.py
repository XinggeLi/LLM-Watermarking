from argparse import Namespace
import yaml
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from watermark_processor import load_model, generate, detect

prompt = "write some code in python"

arg_dict = {
    'model_name_or_path': 'facebook/opt-1.3B',
    'load_fp16': False,
    'prompt_max_length': None,
    'max_new_tokens': 200,
    'generation_seed': 123,
    'use_sampling': True,
    'n_beams': 1,
    'sampling_temp': 0.7,
    'seeding_scheme': 'simple_1',
    'gamma': 0.25,
    'delta': 2.0,
    'normalizers': '',
    'z_threshold': 4.0,
    'ignore_repeated_bigrams': False  # not used by now
}

args = Namespace()
args.__dict__.update(arg_dict)
args.normalizers = args.normalizers.split(",") if args.normalizers else []

model, tokenizer, device = load_model(args)

_, _, output_wo_watermark, output_w_watermark, _ = generate(prompt, args, model, device, tokenizer)

result_wo_watermark, _ = detect(output_wo_watermark, args, device=device, tokenizer=tokenizer)
result_w_watermark, _ = detect(output_w_watermark, args, device=device, tokenizer=tokenizer)

def convert_numpy_to_python(data):
    if isinstance(data, dict):
        return {k: convert_numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(i) for i in data]
    elif isinstance(data, np.float64):  # or np.float32 if that's what you're using
        return float(data)
    else:
        return data

data = {
    "input": {
        "device": device,
        "args": arg_dict,
        "prompt": prompt
    },
    "output": {
        "wo_watermark": {
            "text": output_wo_watermark,
            "metrics": convert_numpy_to_python(result_wo_watermark)
        },
        "w_watermark": {
            "text": output_w_watermark,
            "metrics": convert_numpy_to_python(result_w_watermark)
        }
    }
}

yaml_output = yaml.dump(data, sort_keys=False, default_flow_style=False)

print(yaml_output)

file_path = os.path.join(current_dir, "trial_output.yaml")
with open(file_path, "w") as file:
    yaml.dump(data, file, sort_keys=False, default_flow_style=False)
