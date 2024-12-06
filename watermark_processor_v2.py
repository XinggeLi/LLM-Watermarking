import torch
from transformers import LogitsProcessor
from math import sqrt
from scipy.stats import norm


# seeding_scheme = "simple_1"
# This is what they used in the demo, they implemented several
# versions in the extended processor. I think we don't need to
# implement other seeding schemes for now.

# rng = None
# This is for lazy init so that the rng can depend on the llm model's device

# hash_key = 15485863
# This can be any large prime number.

class WatermarkBase:
    def __init__(
            self,
            vocab=None,
            gamma=0.5,
            delta=2.0,
            seeding_scheme="simple_1",
            hash_key=15485863
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key

    def _seed_rng(self, input_ids):
        if self.seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme: {self.seeding_scheme} requires at least an 1 token prefix sequence to seed rng"
            seed = self.hash_key * input_ids[-1].item()
            self.rng.manual_seed(seed)
        else:
            raise NotImplementedError(f"not implemented seeding_scheme: {self.seeding_scheme}")

    def _get_green_ids(self, input_ids):
        self._seed_rng(input_ids)

        vocab_perm = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        green_ids = vocab_perm[:int(self.gamma * self.vocab_size)]

        return green_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_logits(self, logits, green_ids_batch):
        green_mask = torch.zeros_like(logits, dtype=torch.bool)
        for b_idx, green_ids in enumerate(green_ids_batch):
            green_mask[b_idx, green_ids] = True

        logits[green_mask] += self.delta
        return logits

    def __call__(self, input_ids, logits):
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        green_ids_batch = [self._get_green_ids(input_ids[batch_idx]) for batch_idx in range(input_ids.shape[0])]

        logits = self._process_logits(logits, green_ids_batch)
        return logits
    
class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device = None,
        z_threshold = 4.0,
        normalizers = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.ignore_repeated_bigrams = ignore_repeated_bigrams  # not use by now

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids,
        return_num_tokens_scored = True,
        return_num_green_tokens = True,
        return_green_fraction = True,
        return_green_token_mask = False,
        return_z_score = True,
        return_p_value = True,
    ):
        num_tokens_scored = len(input_ids) - self.min_prefix_len
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                )
            )
        # Standard method.
        # Since we generally need at least 1 token (for the simplest scheme)
        # we start the iteration over the token sequence with a minimum
        # num tokens as the first prefix for the seeding scheme,
        # and at each step, compute the greenlist induced by the
        # current prefix and check if the current token falls in the greenlist.
        green_token_count, green_token_mask = 0, []
        for idx in range(self.min_prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self._get_green_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_mask.append(True)
            else:
                green_token_mask.append(False)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def detect(
        self,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        # call score method
        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict