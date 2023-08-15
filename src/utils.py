import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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

def set_seed(seed):
    """
    Sets the seed for generating random numbers for Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
                
class MMDataset(Dataset):
    def __init__(self, root_dir='./', data_file='data/stils/data_stils.csv', split=None, use_rand_splits=True, rand_seed=42):
        """
        Args:
            root_dir (string): path to data directory
            csv_file (string): Path to the file w/ the dataset information.
            split (string): 'train', 'val', or 'test'.
            use_rand_splits (bool): Whether to use random splits or fixed splits.
            rand_seed (int): Random seed to use for splitting the dataset.
        """
        self.root_dir = root_dir
        df = pd.read_csv(data_file)
        
        # Split the dataset into train+val set and test set
        if use_rand_splits:
            train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=rand_seed)
        else:
            train_val_df, test_df = df[df['split'] == 'train'], df[df['split'] == 'test']
        # Further split the train+val set into separate training and validation sets
        train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=rand_seed)  # 0.11 x 0.9 = 0.099 ~ 0.1

        self.df = train_df if split == 'train' else val_df if split == 'val' else test_df if split == 'test' else df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Filter the dataframe based on the split

        # Load the tensors
        wsi_feats_path = os.path.join(self.root_dir, self.df.iloc[idx]['wsi_feat_path'])
        report_feats_path = os.path.join(self.root_dir, self.df.iloc[idx]['report_feat_path'])
        wsi_feats = torch.load(wsi_feats_path)
        # avg wsi feats along spatial dims
        # wsi_feats = torch.mean(wsi_feats, dim=(1,2))
        report_feats = torch.load(report_feats_path)
        # concat wsi and report feats
        # mm_feats = torch.cat((wsi_feats, report_feats.squeeze()))
        stil_score = self.df.iloc[idx]['stil_score']
        stil_level = self.df.iloc[idx]['stil_lvl']
        
        return wsi_feats, report_feats, stil_score, stil_level
        
    def mm_collate_fn(batch):
        wsi_feats, report_feats, stil_scores, stil_levels = zip(*batch)
        return list(wsi_feats), list(report_feats), default_collate(stil_scores).float(), default_collate(stil_levels)
    
def classify_subtype_grade_zs(lm_name, reports_dir):
    """
    Classify cancer subtypes and grade using zero-shot classification.

    Parameters:
    - lm_name (str): Name of the language model to use.
    - reports_dir (str): Directory containing the report files.

    Returns:
    - DataFrame: A dataframe with report names as indices and predicted labels for each category as columns.
    """
    
    # Load LM & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    lm = AutoModelForSequenceClassification.from_pretrained(lm_name)

    # Define the labels for each subtype & grade
    region_labels = ['ductal/intraductal', 'lobular', 'metastatic', 'other'] # region of origin
    localization_labels = ['in-situ', 'invasive/infiltrating', 'metastatic', 'other'] # degree of localization
    grade_labels = ['grade 1', 'grade 2', 'grade 3', 'NA'] # grade

    # Create a zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model=lm, tokenizer=tokenizer)

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

        # The predicted subtype for each category will be the label with the highest aggregated score
        predicted_region = max(region_scores, key=region_scores.get)
        predicted_localization = max(localization_scores, key=localization_scores.get)
        predicted_grade = max(grade_scores, key=grade_scores.get)

        # Append results
        results.append([report.replace('.txt', ''), predicted_region, predicted_localization, predicted_grade])

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=['case_id', 'region', 'localization', 'grade'])
    df.set_index('case_id', inplace=True)

    return df
