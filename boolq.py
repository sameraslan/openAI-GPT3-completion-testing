import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import argparse
import subprocess
import os
import openai

class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        input_encoding = question + " [SEP] " + passage

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(answer, dtype=torch.long)  # labels are the answers (yes/no)
        }


def evaluate_model(model, dataloader, device):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    return dev_accuracy.compute()

def openAICall(model, prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.4,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response

def q4():
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data
    
    dataset_train_subset = dataset['train'][:100]

    print("Size of the loaded dataset:")
    print(f" - train: {len(dataset_train_subset['passage'])}")


    passages = list(dataset_train_subset['passage'])
    questions = list(dataset_train_subset['question'])
    answers = list(dataset_train_subset['answer'])

    # Put in for loop 30 times and call api each time

    listOfPassages = []
    listOfQuestions = []
    listOfAnswers = []
    basePrompt = ""

    previousAnswer = True
    iterator = 0


    # # Create list of p, q, a, s.t. answers alternate between true and false
    while len(listOfAnswers) < 8:
        if answers[iterator] != previousAnswer:
            listOfAnswers.append(answers[iterator])
            listOfPassages.append(passages[iterator])
            listOfQuestions.append(questions[iterator])
            
            # Build prompt w 8 demonstrations
            basePrompt += "Passage: " + passages[iterator] + "\n"
            basePrompt += "Question: " + questions[iterator] + "\n"
            basePrompt += str(answers[iterator]) + "\n"
            basePrompt += "\n"

            previousAnswer = answers[iterator]
        
        iterator += 1

    print(basePrompt)
    numCorrect = 0
    numInstances = 30

    # Now loop over next 30 instances and feed into api
    for i in range(numInstances):
        prompt = basePrompt + "Passage: " + passages[iterator + i] + "\n" + "Question: " + questions[iterator + i] + "\n"
        print(prompt)
        completion = openAICall("davinci", prompt)
        print("Completion:", completion["choices"][0]["text"])
        print("Answer:", str(answers[iterator + i]))

        if str(completion["choices"][0]["text"]) == str(answers[iterator + i]):
            numCorrect += 1

    print("\n")
    print("\nCorrectly Labelled:", numCorrect)
    print("Accuracy:", numCorrect / numInstances)

# the entry point of the program
if __name__ == "__main__":
    q4()