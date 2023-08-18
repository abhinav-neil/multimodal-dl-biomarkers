import os
import random
import re
import time
from token import OP
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline, AutoConfig

def extract_text_from_pdf(data_dir):
    """
    This function extracts text from PDF files in a given directory and its subdirectories.
    It saves the extracted text as a .txt file with the same name as the original PDF file.
    
    Args:
    data_dir (str): The directory to search for PDF files.
    """
    # Loop through all the folders in the current directory
    for root, dirs, files in tqdm(os.walk(data_dir)):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            
            # Check if the file is a PDF
            if file.endswith('.PDF'):
                # Construct the new file path with a lowercase extension
                new_file_path = file_path[:-4] + '.pdf'
                
                # Check if a file with the new name already exists
                if os.path.exists(new_file_path):
                    # If it does, delete the original file
                    os.remove(file_path)
                else:
                    # If it doesn't, rename the original file
                    os.rename(file_path, new_file_path)
                
                # Update the file path to reflect the new name
                file_path = new_file_path
            
            # Check if the file is a PDF (again, in case it was just renamed)
            if file.endswith('.pdf'):
                # Construct the path for the text file
                text_file_path = file_path.replace('.pdf', '.txt')
                
                # Check if the text file already exists
                if not os.path.exists(text_file_path):
                    # If it doesn't, create a PDF file reader object
                    reader = PdfReader(file_path)
                    
                    # Initialize an empty string to hold the text
                    text = ''
                    
                    # Loop through all the pages in the PDF file
                    for page in reader.pages:
                        # Extract the text from the page
                        page_text = page.extract_text()
                        
                        # Append the page text to the overall text
                        text += page_text
                    
                    # Write the extracted text to a file
                    with open(text_file_path, 'w') as output_file:
                        output_file.write(text)
      
def distill_reports(reports_dir, summary_dir, res_dir):
    # Ensure output directories exist
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Loop through all the reports in the reports_dir
    for report_file in os.listdir(reports_dir):
        if report_file.endswith('.txt'):
            with open(os.path.join(reports_dir, report_file), 'r') as f:
                report = f.read()

                # Extract relevant sentences for subtype and grade
                keywords = ['carcinoma', 'grade']
                # This regex will match any sentence containing any of the keywords
                pattern = r'([^.!?\n]*\b(?:' + '|'.join(keywords) + r')\b[^.!?]*[.!?\n])'
                rel_sents = re.findall(pattern, report, re.IGNORECASE)

                # Combine the extracted sentences
                summary = ' '.join(rel_sents)
                # check if summary is empty, if so, skip
                if summary == '':
                    continue
                
                # get residual = report - summary
                res = report.replace(summary, '')
                
                # save the summary and residual
                with open(os.path.join(summary_dir, report_file), 'w') as f_summary:
                    f_summary.write(summary)
                
                with open(os.path.join(res_dir, report_file), 'w') as f_res:
                    f_res.write(res)
                                      
def extract_text_features(lm, tokenizer, data_dir, output_dir):
    """
    Function to extract features from text reports using BioBERT.

    Args:
    lm (transformers.PreTrainedModel): The pretrained BioBERT model.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to preprocess the reports.
    data_dir (str): Path to the directory containing the case folders.
    output_dir (str): Path to the directory to save the extracted features.
    
    Returns:
    None. The function saves the extracted features to .pt files in the 'report_feats' directory.
    """

    # create directory to save extracted features
    # output_dir = f'{data_dir}/report_feats'
    os.makedirs(output_dir, exist_ok=True)

    # loop through each case folder (folders starting with 'TCGA-')
    for case_folder in os.listdir(data_dir):
        if case_folder.startswith('TCGA-'):
            case_folder_path = os.path.join(data_dir, case_folder)

            # loop through each file in the case folder
            for filename in tqdm(os.listdir(case_folder_path)):
                # check if it's a .txt file
                if filename.endswith('.txt'):
                    file_path = os.path.join(case_folder_path, filename)

                    # open the file and read its content
                    with open(file_path, 'r') as file:
                        report = file.read()

                    # preprocess/tokenize the report
                    inputs = tokenizer(report, return_tensors='pt', padding=True, truncation=True, max_length=512)

                    # extract features using the pretrained biobert model
                    with torch.no_grad():
                        outputs = lm(**inputs)

                    # get the hidden states of the last layer
                    last_hidden_states = outputs.last_hidden_state

                    # compute the mean of the hidden states
                    report_feats = torch.mean(last_hidden_states, dim=1)

                    # save the extracted features
                    report_feats_filename = f'{filename[:-4]}.report.pt'  # remove '.txt' from filename
                    report_feats_file_path = os.path.join(output_dir, report_feats_filename)
                    torch.save(report_feats, report_feats_file_path)

                    break
                
def classify_subtype_grade_zs(lm_name, reports_dir):
    """
    Classify cancer subtypes and grade using zero-shot classification.

    Parameters:
    - lm_name (str): Name of the language model to use.
    - reports_dir (str): Directory containing the report files.

    Returns:
    - DataFrame: A dataframe with report names as indices and predicted labels for each category as columns.
    """
    # check if cuda is availabled
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration and set gradient checkpointing
    config = AutoConfig.from_pretrained(lm_name)
    # config.gradient_checkpointing = True

    # Load LM & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    # with init_empty_weights():
    #     empty_model = AutoModelForSequenceClassification.from_config(config)
        
    # # get device map
    # device_map = infer_auto_device_map(empty_model)
    
    # print('device_map:', device_map)
    
    lm = AutoModelForSequenceClassification.from_pretrained(lm_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    # lm = AutoModelForSequenceClassification.from_pretrained(lm_name, torch_dtype=torch.float16, device_map=device_map, offload_folder='offload', low_cpu_mem_usage=True)
    
    # transfer model to device
    # lm.to(device)
    
    # Define the labels for each subtype & grade
    region_labels = ['ductal/intraductal', 'lobular', 'other'] # region of origin
    localization_labels = ['in-situ', 'invasive/infiltrating', 'metastatic'] # degree of localization
    grade_labels = ['histologic/nottingham grade 1 (well differentiated)', 'histologic/nottingham grade 2 (moderately differentiated)', 'histologic/nottingham grade 3 (poorly differentiated)', 'None'] # grade

    # Create a zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model=lm, tokenizer=tokenizer, device='cpu')

    # Placeholder for results
    results = []

    # Loop through each report in the directory
    reports_list = sorted(os.listdir(reports_dir))
    for report in tqdm(reports_list):
        report_path = os.path.join(reports_dir, report)
        
        with open(report_path, 'r') as f:
            sample_report = f.read()

        # Tokenize the report and split it into overlapping chunks
        max_length = lm.config.max_position_embeddings - 2  # account for [CLS] and [SEP] tokens
        overlap = 100
        tokens = tokenizer.tokenize(sample_report)
        chunk_size = max_length - overlap - 2  # account for [CLS] and [SEP] tokens and overlap
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size - overlap)]

        # Initialize scores for each label
        region_scores = {label: 0 for label in region_labels}
        localization_scores = {label: 0 for label in localization_labels}
        grade_scores = {label: 0 for label in grade_labels}

        # Classify each chunk and aggregate the results
        for chunk in chunks:
            chunk_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(chunk))
            
            # Classify for region
            region_result = classifier(chunk_text, region_labels)
            for label, score in zip(region_result["labels"], region_result["scores"]):
                region_scores[label] += score
                
            # Classify for localization
            localization_result = classifier(chunk_text, localization_labels)
            for label, score in zip(localization_result["labels"], localization_result["scores"]):
                localization_scores[label] += score
                
            # Classify for grade
            grade_result = classifier(chunk_text, grade_labels)
            for label, score in zip(grade_result["labels"], grade_result["scores"]):
                grade_scores[label] += score

        # agg preds per category
        predicted_region = max(region_scores, key=region_scores.get)
        # predicted_region = max(region_scores, key=region_scores.get)
        # localization priority: metastatic > invasive/infiltrating > in-situ
        predicted_localization = 'metastatic' if localization_scores['metastatic'] > 0.2 else 'invasive/infiltrating' if localization_scores['invasive/infiltrating'] > 0.3 else 'in-situ'
        predicted_grade = max(grade_scores, key=grade_scores.get)

        # Append results
        results.append([report.replace('.txt', ''), predicted_region, predicted_localization, predicted_grade])

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=['case_id', 'region', 'localization', 'grade'])
    df.set_index('case_id', inplace=True)

    return df

def gen_subtype_grade_zs(lm_name, prompt, api=None, args={}):
    """
    Generate subtype and grade using a language model.

    Args:
    - lm_name (str): Name of the language model to use.
    - prompt (str): The prompt for the generative model.
    - api (str): type of API to use, one of 'hf' (HuggingFace) or 'openai' (OpenAI) or None (local).
    - args: Additional arguments for the model or API.

    Returns:
    - str: Generated response.
    """
    
    # Load environment variables
    load_dotenv()
    HF_API_KEY = os.getenv("HF_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if api == 'hf':
        
        if not HF_API_KEY:
            raise ValueError("HF_API_KEY not found in environment variables. Please set it up.")
        
        # Use HuggingFace's Inference API
        API_URL = f"https://api-inference.huggingface.co/models/{lm_name}"
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        data = {
            "inputs": prompt,
            **args  # Add any additional arguments
        }
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()[0]['generated_text']
    
    elif api == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        API_URL = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": lm_name,
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}],
            **args  # Add any additional arguments
        }
        
        # Retry mechanism
        max_retries = 3
        wait_time = 10  # in seconds
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.HTTPError as e:
                print(f"HTTP error: {e}, retrying...")
                if attempt < max_retries - 1:  # i.e. if it's not the last attempt
                    time.sleep(wait_time)  # wait for a bit before retrying
                else:
                    print("max retries reached, skipping...")
                    break  # exit the loop and move to the next report
                    

    else:
        # Load the model and tokenizer locally
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        model = AutoModelForCausalLM.from_pretrained(lm_name)

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate a response
        output = model.generate(input_ids, **args)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text

def create_zs_prompt(report_path):
    """
    Create a prompt for zero-shot generation of grade and subtype from report text.
    
    Args:
    - report_path (str): Path to the report file.
     
    Returns:
    - prompt (str): The prompt for zero-shot generation.
    """
    
    with open (report_path, 'r') as f:
        report = f.read()
    
    prompt = f'''Read the following report: \n\n{report}\n\nBased on this info, fill in the following fields below. For each field, choose the correct answer from the given choices. YOUR ANSWER SHOULD EXACTLY MATCH ONE OF THE PROVIDED CHOICES W/O ANY VARIATIONS! Choose 'NA' if none of the other options are correct, or if the report contains insufficient info. Your answer should always stick to the template provided below. \nRegion of occurrence [ductal, lobular, other (specify), NA] ('ductal' and 'intraductal' are equivalent): \nDegree of localization: [in situ, invasive, metastatic, NA] ('invasive' and 'infiltrating' are equivalent. If more than one, select the highest priority, i.e., metastatic > invasive > in situ): \nHistological/Nottingham grade: [1, 2, 3, NA] (1: well differentiated, 2: moderately differentiated, 3: poorly differentiated): '''
    
    return prompt

def classify_reports_zs(lm_name, reports_dir, api='openai', results_path='data/data_subtype_grade_annot.csv', args={}):
    """
    Classify reports using zero-shot generation.

    Args:
    - lm_name (str): Name of the language model to use.
    - reports_dir (str): Directory containing the report files.
    - api (str): API to use for inference, one of 'hf', 'openai', or None for local.
    - results_path (str): Path to save the classification results.
    - args (dict): Additional arguments for the model or API.

    Returns:
    - DataFrame: Pandas DataFrame containing the classification results.
    """
    
    # List to store results
    results = []
    
    # Loop through all reports in the directory
    for report_file in tqdm(os.listdir(reports_dir)):
        if report_file.endswith('.txt'):
            # Extract case_id from filename
            case_id = os.path.splitext(report_file)[0]
            
            # Create prompt for the report
            report_path = os.path.join(reports_dir, report_file)
            prompt = create_zs_prompt(report_path)
            
            # Generate output using the model
            output = gen_subtype_grade_zs(lm_name, prompt, api=api, args=args)
            
            # Extract values from the generated output
            # First, split by double line breaks
            lines = output.split('\n\n')
            # If there's only one item, it means there were no double line breaks, so split by single line breaks
            if len(lines) == 1:
                lines = output.split('\n')
                
            try:
                region = lines[0].split(': ')[1].strip()
                localization = lines[1].split(': ')[1].strip()
                grade = lines[2].split(': ')[1].strip()
            
            except IndexError:
                # If the output is malformed, set all values to 'NA'
                print(f'error parsing output for {report_file}...')
                region, localization, grade = 'NA', 'NA', 'NA'
                
            # Append results to the list
            results.append({
                "case_id": case_id,
                "region": region,
                "localization": localization,
                "grade": grade
            })
    
    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    # sort by case_id
    df.sort_values(by=['case_id'], inplace=True)
    df.to_csv(results_path, index=False)
    
    return df