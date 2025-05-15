import read_data


def bert_sentence_inference_s1(sentences, device='cuda:0'):
    '''
    this is to use bert-base-nli-cls-token to generate sentence embedding
    '''
    from transformers import AutoTokenizer, AutoModel
    import torch

    def cls_pooling(model_output, attention_mask):
        return model_output[0][:,0]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('llm/bert-base-nli-cls-token')
    model = AutoModel.from_pretrained('llm/bert-base-nli-cls-token').to(device)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings


def bert_sentence_inference_s2(sentences, device='cuda:0'):
    '''
    thi is to use all-MiniLM-L6-v1 to generate sentence embedding
    '''
    from transformers import AutoTokenizer, AutoModel
    import torch

    def cls_pooling(model_output, attention_mask):
        return model_output[0][:,0]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('llm/all-MiniLM-L6-v1')
    model = AutoModel.from_pretrained('llm/all-MiniLM-L6-v1').to(device)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()

    return sentence_embeddings


def bert_sentence_inference_s3(sentences, batch_size = 32, device='cuda:0'):#embedding dimension 768
    '''
    this is to use biobert to generate sentence embedding
    '''
    #def sentence_embedding(sentences, model_path = 'bert_mlm/', batch_size = 32, device='cuda:0'):#embedding dimension 768
    from transformers import BertTokenizer, BertModel
    from tqdm import tqdm
    import torch
    #import torch
    model_path = 'llm/bioBERT_v1.1'
    print('sentence num:',len(sentences))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path).to(device)
    print('load model and tokenizer succesful')

    # Tokenize the batch of sentences with the specified batch size
    batched_inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    print('sentences token generated')

    num_batches = len(batched_inputs["input_ids"]) // batch_size
    remaining_elements = len(batched_inputs["input_ids"]) % batch_size
    all_cls_embeddings = []
    for i in tqdm(range(num_batches + (1 if remaining_elements > 0 else 0))):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(batched_inputs["input_ids"]))
        # Extract batch
        batch_inputs = {key: value[start_idx:end_idx] for key, value in batched_inputs.items()}
        outputs = model(**batch_inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].tolist()
        all_cls_embeddings += cls_embeddings
    print(f'embedding inference finished with {len(all_cls_embeddings)} embeddings with dimension {len(all_cls_embeddings[0])}')
    return all_cls_embeddings


def bert_sentence_inference_s4(sentences, device='cuda:0'):
    '''
    thi is to use all-MiniLM-L6-v1 to generate sentence embedding
    '''

    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F


    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    # Sentences we want sentence embeddings for
    #sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('llm/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('llm/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    #print(sentence_embeddings.shape)
    return sentence_embeddings


def side_effect_description_inference(device='cuda:0'):
    side_effects = read_data.get_all_side_descriptions()
    return bert_sentence_inference(side_effects, device)


def drug_description_inference(device='cuda:0'):
    drug_descriptions = read_data.get_all_drug_descriptions()
    return bert_sentence_inference(drug_descriptions, device)

def mistral_inference(device='cuda:0'):
    """
    from transformers import pipeline

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        ]
    chatbot = pipeline("text-generation", model="llm/Mistral-7B-Instruct-v0.3", max_new_tokens=30, device = device)
    chatbot(messages)

    this do work
    """
    print("employed device:", device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import time
    # Clear GPU cache
    #torch.cuda.empty_cache()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("llm/Mistral-7B-Instruct-v0.3", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("llm/Mistral-7B-Instruct-v0.3")

    #prompt = "abdominal discomfort is a common side effect caused by molecues with certain strcutures. please write the top three sub-structure SMILES format seperated by comma"
    #prompt = "abdominal discomfort is a common side effect caused by molecues with certain strcutures. please write the top 1 sub-structure SMILES format"
    #prompt = "You are a biochemistry expert.\n <dataset>:drug side-effect frequency \n <task>: drug side-effect prediction \n <side effect> abdominal discomfor \n <query> the possible sub-structure of molecure that could cause abdominal discomfort"
    
    prompt = "You are a drug development expert.\n "
    prompt += "<dataset>:drug side-effect frequency \n "
    prompt += "<side effect>: abdominal discomfort \n "
    prompt += "<side effect description>: Abdominal discomfort is mild pain or unease in the stomach area. It may arise from indigestion, gas, or inflammation. \n"
    prompt += 'please provide the possible top fiave sub-structure of molecule in SMILES format that could cause abdominal discomfort in the following formats: \n'
    #prompt += '<sub-strcuture SMILES 1>:<s_begin>...<s_end><EXPLAINE1>...<sub-strcuture SMILES2>:<s_begin>...<s_end><EXPLAINE2>...<sub-strcuture SMILES3>:<s_begin>...<s_end><EXPLAINE3>...<sub-strcuture SMILES4>:<s_begin>...<s_end><EXPLAINE4>...<sub-strcuture SMILES5>:<s_begin>...<s_end><EXPLAINE5>...'
    prompt += '<sub-strcuture SMILES 1>:...<EXPLAINE1>...<sub-strcuture SMILES2>:...<EXPLAINE2>...<sub-strcuture SMILES3>:...<EXPLAINE3>...<sub-strcuture SMILES4>:...<EXPLAINE4>...<sub-strcuture SMILES5>:...<EXPLAINE5>...'
    #prompt += '<sub-strcuture SMILES 1>:...<sub-strcuture SMILES2>:...<sub-strcuture SMILES3>:...<sub-strcuture SMILES4>:...<sub-strcuture SMILES5>:...'
    
    print("move tokenizer to ", device)
    # Tokenize input
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    #print("move model to ", device)
    # Move model to device
    #model.to(device)

    print("start to generate")

    # Use mixed precision
    #with torch.cuda.amp.autocast():
    start_time = time.time()

    generated_ids = model.generate(**model_inputs, max_new_tokens=500, do_sample=True)

    end_time = time.time()

    print("time:", end_time - start_time)

    result = tokenizer.batch_decode(generated_ids)[0]
    print('generated content', result)


def mistral_inference_prompt(device='cuda:0', prompt = 'hi'):
    """
    from transformers import pipeline

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        ]
    chatbot = pipeline("text-generation", model="llm/Mistral-7B-Instruct-v0.3", max_new_tokens=30, device = device)
    chatbot(messages)

    this do work
    """
    print("employed device:", device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import time
    # Clear GPU cache
    #torch.cuda.empty_cache()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("llm/Mistral-7B-Instruct-v0.3", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("llm/Mistral-7B-Instruct-v0.3")

    #prompt = "abdominal discomfort is a common side effect caused by molecues with certain strcutures. please write the top three sub-structure SMILES format seperated by comma"
    #prompt = "abdominal discomfort is a common side effect caused by molecues with certain strcutures. please write the top 1 sub-structure SMILES format"
    #prompt = "You are a biochemistry expert.\n <dataset>:drug side-effect frequency \n <task>: drug side-effect prediction \n <side effect> abdominal discomfor \n <query> the possible sub-structure of molecure that could cause abdominal discomfort"
    
    prompt = prompt
    
    print("move tokenizer to ", device)
    # Tokenize input
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    #print("move model to ", device)
    # Move model to device
    #model.to(device)

    print("start to generate")

    # Use mixed precision
    #with torch.cuda.amp.autocast():
    start_time = time.time()

    generated_ids = model.generate(**model_inputs, max_new_tokens=500, do_sample=True)

    end_time = time.time()

    print("time:", end_time - start_time)

    result = tokenizer.batch_decode(generated_ids)[0]
    print('generated content', result)


def mistral_inference_not_work(device='cuda:0'):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("employed device:", device)

    model = AutoModelForCausalLM.from_pretrained("llm/Mistral-7B-Instruct-v0.3", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("llm/Mistral-7B-Instruct-v0.3")

    prompt = "My favourite condiment is"

    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    model.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True)
    result = tokenizer.batch_decode(generated_ids)[0]
    print(result)

def mistral_inference_with_transformers():
    #this is on CPU, not ideal
    #and this do not work
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModel
    import torch
    messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
    ]  
    chatbot = pipeline("text-generation", model="llm/Mistral-7B-Instruct-v0.3", max_length = 200)
    chatbot(messages)

def chemberta_smiles_inference(smiles_list, device='cuda:0'):
    """
    This function uses chemberta-77m-mrm to generate embeddings for a list of SMILES strings.
    """

    # Load model and tokenizer from HuggingFace Hub
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('llm/ChemBERTTa-77M-MLM')
    model = AutoModel.from_pretrained('llm/ChemBERTTa-77M-MLM').to(device)

    '''
    encoded_input = tokenizer(
    smiles_list, 
    padding=True, 
    truncation=True, 
    max_length=128,  # Set the maximum length of tokenised sequences
    return_tensors='pt'
    ).to(device)
    '''

    # Tokenize SMILES strings
    encoded_input = tokenizer(smiles_list, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, we use the CLS token representation
    embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

    return embeddings

def chemberta_smiles_inference_entire_sequence(smiles_list, device='cuda:0'):
    """
    This function uses chemberta-77m-mrm to generate embeddings for a list of SMILES strings.
    """

    # Load model and tokenizer from HuggingFace Hub
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('llm/ChemBERTTa-77M-MLM')
    model = AutoModel.from_pretrained('llm/ChemBERTTa-77M-MLM').to(device)

    '''
    encoded_input = tokenizer(
    smiles_list, 
    padding=True, 
    truncation=True, 
    max_length=128,  # Set the maximum length of tokenised sequences
    return_tensors='pt'
    ).to(device)
    '''

    #print(smiles_list[0])

    # Tokenize SMILES strings
    encoded_input = tokenizer(smiles_list, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, we use the CLS token representation

    # Get all hidden states for all tokens and convert to numpy array
    embeddings = model_output.last_hidden_state.cpu().numpy()
    #embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

    return embeddings

def demo_inference():
    user_prompt = 'tell me about drug-side effect frequency association'
        
    encodeds = tokenizer(user_prompt, return_tensors="pt")
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=300, do_sample=True)
    # decode with mistral tokenizer
    result = tokenizer.decode(generated_ids[0].tolist())

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write('{}\t{}\n'.format(dataset, result))

def drug_strcuture_llm():
    pass



if __name__ == "__main__":
    # Example usage with GPU
    # side_effect_description_inference(device='cuda')
    #drug_description_inference(device='cuda')
    '''
    prompt = "you are a drug development expert.\n"
    prompt += "<side effect>:abdominal discomfort\n"
    prompt += "please provide the possible cause of abdominal discomfort in the following formats: \n"
    prompt += "<cause>...<cause>...<cause>"

    prompt = 'possible cause of the side effect: <abdominal discomfor>'
    mistral_inference_prompt(prompt=prompt)
    '''

    #biobert_inference(['abdominal discomfort','internal discharge'])
    chemberta_smiles_inference(['CCO', 'CCN', 'CCOCC', 'CCNCC', 'CCOCCC', 'CCNCCC'])