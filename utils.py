import torch

# function to compute log-probs of y (reply) | x (prompt). No Gemma!
def CLRL(model, tok, prompts, replies, MAX_LEN=1024, device=None):

    '''
    Args:
    model - AutoModelforCausalLM object
    tok - AutoTokenizer object
    prompts - question that we are asking (list of strings)
    replies - answer that we received (list of strings)
    device - returned tensors are on that device (string)

    Returns:
    sum_logprob_replies - the log-probabilities of the reply given the prompt (size-B tensor)
    length_replies - the number of tokens in the reply (size-B tensor)
    '''
    # 0. get the batch-size
    batch_size = len(prompts)

    if device is None:
        device = next(model.parameters()).device

    # 1. convert ONLY THE PROMPT into a format for this LLM (STILL STRING!)
    prompt_only = tok.apply_chat_template(

        # a. store prompt as user-specified.
        [[{"role" : "user", "content" : prompt}] for prompt in prompts],

        # b. keep it as a string (as opposed to tokens)
        tokenize=False,

        # c. append an assistant header so the model knows it's time to speak.
        add_generation_prompt=True
    )

    # 2. convert PROMPT + REPLY into format for this LLM (STILL STRING)
    prompts_and_replies = tok.apply_chat_template(

        # a. store prompt + reply as user-specified
        [[{"role" : "user", "content" : prompt},
        {"role" : "assistant", "content" : reply}] for (prompt, reply)
        in zip(prompts, replies)],

        # b. still keep it as a string (as opposed to tokens)
        tokenize=False,

        # c. we don't need an assistant header anymore because no more comms.
        add_generation_prompt=False
    )

    # 3. convert the prompt itself into token_ids and attention masks
    tokens_prompt = tok(

        # a. return prompt as a PyTorch tensor of token-ids.
        prompt_only, return_tensors="pt",

        # b. truncate the prompt if way too long w.r.t. max_length
        truncation=True, max_length=MAX_LEN,

        # c. already added assistant-headers / etc. earlier. No more specials.
        add_special_tokens=False,

        # d. need to add padding for the batching
        padding=True
    )

    # 4. repeat for PROMPT + REPLY
    tokens_prompts_and_replies = tok(
        prompts_and_replies, return_tensors="pt", truncation=True, max_length=MAX_LEN,
        add_special_tokens=False, padding=True
    )

    # 5. move the tokens for the PROMPT + REPLY to the model's device. Also the mask because we're batching!
    input_ids = tokens_prompts_and_replies.input_ids.to(device)
    attention_mask = tokens_prompts_and_replies.attention_mask.to(device)

    # 6. get the lengths of each prompt (no. of non-padding tokens) per example (prompt, prompt+reply)
    prompt_lengths = tokens_prompt.attention_mask.sum(dim=-1) # B numbers.
    pairs_lengths = tokens_prompts_and_replies.attention_mask.sum(dim=-1) # B numbers.

    '''
    Note: if we truncate for limited-compute by MAX_LEN due to compute, we could potentially only have prompt / part of prompt. The "min" is to hedge!
    '''
    # 7. find where the LLM's responses start + ends
    response_starts = torch.minimum(prompt_lengths, pairs_lengths) # B numbers.
    response_ends = pairs_lengths # B numbers, too.

    # 8. get the logits per token-id (at t for t+1).
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # [B, L, V]

    '''
    # 9. convert to log-probs via log-softmax over the vocabulary axis (-1)
    logprobs_all = logits.log_softmax(-1) # [B, L, V]
    '''

    # 10. instantiate lists of our sum-logprobs and response lengths
    sum_logprob_replies = []
    length_replies = []

    # 11. process each example in the batch separately
    for i in range(batch_size):

        # a. compute the particular response_start and response_end
        response_start = int(response_starts[i].item())
        response_end = int(response_ends[i].item())

        # b. if the edge case of horrible truncation happens, special returns.
        if response_end <= response_start:
            sum_logprob_replies.append(None)
            length_replies.append(0)
            continue

        # c. which token_ids are we trying to "predict"? [L]
        target_input_ids = input_ids[i, response_start : response_end]

        # d. what are the pred. probabilities 1-step before these target tokens?
        # pred_logprobs = logprobs_all[i, max(response_start - 1, 0) : response_end - 1]
        pred_logits = logits[i, max(response_start - 1, 0):response_end - 1, :]  # [L, V]

        # let's take logsumexp over the vocabulary (as a denominator!)
        logZ = torch.logsumexp(pred_logits, dim=-1)  # [L]

        '''
        # e. get the log-probs corresponding to each target input_id: [L, V] -> [L, 1]
        logprobs = pred_logprobs.gather(

            # a. gather the entries for the LAST DIMENSION of pred_logprobs
            -1,

            # b. [1, L] -> [1, L, 1], to have same dims as [1, L, V]
            target_input_ids.unsqueeze(-1)).squeeze(-1) # [1, L, 1] -> [1, L].
        '''

        # pick out logits for the actual tokens
        token_logits = pred_logits.gather(
            -1,
            target_input_ids.unsqueeze(-1)  # [L, 1]
        ).squeeze(-1)  # [L]

        #  log-probs for each reply token by subtracting out the logsumexp denominator.
        logprobs = token_logits - logZ  # [L]

        # f. add to our lists
        sum_logprob_replies.append(logprobs.sum())
        length_replies.append(target_input_ids.numel())

    # 12. return our lists
    return sum_logprob_replies, length_replies

# compute accuracy on an arbitrarily-sized test/val set without overloading GPU.
def compute_acc(model, tok, prompt, chosen, rejected, device, MAX_BATCH=20):

    '''
    Args:
    model - instance of AutoModelCausalLM
    tok - corresponding tokenizer
    prompts - list of str prompts.
    chosen - list of str chosen replies.
    rejected - list of str rejected replies.

    Returns:
    accuracy - float (length-normalized log-probs comparison!)
    '''

    # counters for num correct and total
    correct, total = 0, 0

    # how many minibatches do we need?
    N = len(prompt)
    N_BATCHES = (N // MAX_BATCH) if (N % MAX_BATCH) == 0 else (N // MAX_BATCH) + 1

    # disable gradient tracking
    with torch.no_grad():

        # go thru batches of MAX_BATCH size.
        for batch in range(N_BATCHES):

            # get this mini-batch worth of data
            batch_prompt = prompt[batch * MAX_BATCH : ((batch+1) * MAX_BATCH)]
            batch_chosen = chosen[batch * MAX_BATCH : ((batch+1) * MAX_BATCH)]
            batch_rejected = rejected[batch * MAX_BATCH : ((batch+1) * MAX_BATCH)]

            # compute the logprobs on the chosen w.r.t. the prompt (L-NORMALIZED)
            logprobs_chosen, L_chosen = CLRL(
                model=model, tok=tok, device=device,
                prompts=batch_prompt, replies=batch_chosen)
            logprobs_chosen_mean = torch.stack(logprobs_chosen) / torch.tensor(L_chosen, device=device)

            # compute the logprobs on the rejected w.r.t. the prompt (L-NORMALIZED)
            logprobs_rejected, L_rejected = CLRL(
                model=model, tok=tok, device=device,
                prompts=batch_prompt, replies=batch_rejected)
            logprobs_rejected_mean = torch.stack(logprobs_rejected) / torch.tensor(L_rejected, device=device)

            # update our accuracy counters
            outcomes = logprobs_chosen_mean > logprobs_rejected_mean
            correct += outcomes.sum(dtype=torch.int)
            total += outcomes.shape[0]

    # just return the proportion of correct
    return correct / total

def compute_mrpo_objective(preferred_train_log_probs, nonpreferred_train_log_probs, preferred_ref_log_probs, nonpreferred_ref_log_probs, beta, alphas):
    """
    compute MRPO objective as in the MRPO paper.

    Parameters:
    ------------
    preferred_train_log_probs : torch.Tensor of size B
    nonpreferred_train_log_probs : torch.Tensor of size B
    alphas : torch.Tensor of size B x K
    preferred_ref_log_probs : torch.Tensor of size B x K
    nonpreferred_ref_log_probs : torch.Tensor of size B x K
    beta : float

    Returns:
    ------------
    MRPO objective : torch.Tensor of size 1
    """

    # clamping alphas to not make them 0
    alphas_clamped = torch.clamp(alphas, min=1e-6) # B x K

    # computing -log(\sum_k \alpha_k / \pi_ref_k(y^+ | x))
    preferred_ref_log_probs_combined = -torch.logsumexp(torch.log(alphas_clamped)-preferred_ref_log_probs, dim=1)  # B

    # computing -log(\sum_k \alpha_k / \pi_ref_k(y^- | x))
    nonpreferred_ref_log_probs_combined = -torch.logsumexp(torch.log(alphas_clamped)-nonpreferred_ref_log_probs, dim=1) # B

    # computing log(\pi_{\theta}(y^+ | x)) - log(\pi_{\theta}(y^- | x))
    train_log_prob = preferred_train_log_probs - nonpreferred_train_log_probs

    # computing -log(\sum_k \alpha_k / \pi_ref_k(y^+ | x)) + log(\sum_k \alpha_k / \pi_ref_k(y^- | x))
    ref_log_prob = -preferred_ref_log_probs_combined + nonpreferred_ref_log_probs_combined

    # computing log(\pi_{\theta}(y^+ | x)) - log(\pi_{\theta}(y^- | x)) -log(\sum_k \alpha_k / \pi_ref_k(y^+ | x)) + log(\sum_k \alpha_k / \pi_ref_k(y^- | x))
    logprob = train_log_prob + ref_log_prob

    # computing final MRPO loss !
    return -torch.mean(torch.nn.functional.logsigmoid(beta*logprob))

def compute_mdpo_objective(preferred_train_log_probs, nonpreferred_train_log_probs, preferred_ref_log_probs, nonpreferred_ref_log_probs, beta, alphas):
    """
    compute MDPO objective as in the MRPO paper.

    Parameters:
    ------------
    preferred_train_log_probs : torch.Tensor of size B
    nonpreferred_train_log_probs : torch.Tensor of size B
    alphas : torch.Tensor of size B x K
    preferred_ref_log_probs : torch.Tensor of size B x K
    nonpreferred_ref_log_probs : torch.Tensor of size B x K
    beta : float

    Returns:
    ------------
    MDPO objective : torch.Tensor of size 1

    """

    # clamping alphas to not make them 0
    alphas_clamped = torch.clamp(alphas, min=1e-6) # B x K

    # computing log(\pi_{\theta}(y^+ | x)) - log(\pi_{\theta}(y^- | x))
    train_log_prob = preferred_train_log_probs - nonpreferred_train_log_probs # B
    train_log_prob = train_log_prob[:,None] # B x 1

    # computing log(\pi_ref_k(y^+ | x)) - log(\pi_ref_k(y^- | x))
    ref_log_prob = preferred_ref_log_probs - nonpreferred_ref_log_probs # B x K

    # computing log(\pi_{\theta}(y^+ | x)) - log(\pi_{\theta}(y^- | x)) - (log(\pi_ref_k(y^+ | x)) - log(\pi_ref_k(y^- | x)))
    logprob = train_log_prob - ref_log_prob # B x K

    # computing final MDPO loss
    mdpo = torch.nn.functional.logsigmoid(beta*logprob) * alphas_clamped # B x K
    return -torch.mean(torch.sum(mdpo, dim=1))

def compute_dpo_objective_many_refs(preferred_train_log_probs, nonpreferred_train_log_probs, preferred_ref_log_probs, nonpreferred_ref_log_probs, beta):
    """
    compute MDPO objective as in the MRPO paper.

    Parameters:
    ------------
    preferred_train_log_probs : torch.Tensor of size B
    nonpreferred_train_log_probs : torch.Tensor of size B
    preferred_ref_log_probs : torch.Tensor of size B x K
    nonpreferred_ref_log_probs : torch.Tensor of size B x K
    beta : float

    Returns:
    ------------
    DPO objective per ref model : torch.Tensor of size K

    """

    # computing log(\pi_{\theta}(y^+ | x)) - log(\pi_{\theta}(y^- | x))
    train_log_prob = preferred_train_log_probs - nonpreferred_train_log_probs # B

    # computing log(\pi_ref_k(y^+ | x)) - log(\pi_ref_k(y^- | x))
    train_log_prob = train_log_prob[:,None] # B x 1

    # computing log(\pi_ref_k(y^+ | x)) - log(\pi_ref_k(y^- | x))
    ref_log_prob = preferred_ref_log_probs - nonpreferred_ref_log_probs # B x K

    # computing log(\pi_{\theta}(y^+ | x)) - log(\pi_{\theta}(y^- | x)) - (log(\pi_ref_k(y^+ | x)) - log(\pi_ref_k(y^- | x)))
    logprob = train_log_prob - ref_log_prob # B x K

    # computing all K DPO loss
    mdpo = torch.nn.functional.logsigmoid(beta*logprob) # B x K

    # returning mean over the batch for each ref model
    return -torch.mean(mdpo, dim=0) # K