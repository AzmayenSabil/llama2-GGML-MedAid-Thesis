import os
import torch
import pdfplumber

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"
# Fine-tuned model
new_model = "llama-2-7b-chat-medaid"

# Path to your local dataset PDF file 
pdf_path = "G:\LLM-Model-MedAid-Thesis\Model-2\dataset\combined_conversations_with_scenarios.pdf"

# Extract text from PDF using pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

# Save the extracted text to a local file
output_path = "G:\LLM-Model-MedAid-Thesis\Model-2\output_file.txt"
with open(output_path, "w", encoding="utf-8") as output_file:
    output_file.write(text)

# Now you can load the dataset using the load_dataset function
dataset = load_dataset("text", data_files=output_path, split="train")

# Set up GPU configuration for PyTorch
if device == "cuda":
    torch_device = torch.device("cuda")
    torch.cuda.set_device(0)  # Set the GPU device index as needed
else:
    torch_device = torch.device("cpu")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": device}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)


trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
# Local path to your directory
local_path = "G:\LLM-Model-MedAid-Thesis\Model-2\trained_model"
# Save the model
trainer.model.save_pretrained(local_path)
# Save the tokenizer
trainer.tokenizer.save_pretrained(local_path)


logging.set_verbosity(logging.CRITICAL)

# Initial prompt
prompt = "Be a doctor assistant. And keep questioning one by one to extract symptoms and history of the patient. Don't give advice or ask anything else. Just extract symptoms or history by questioning one question at a time."

# Create a pipeline for text generation using the fine-tuned model and tokenizer
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

# File to save the conversation locally
output_file_path = "G:\LLM-Model-MedAid-Thesis\Model-2\conversation_log\log.txt"  # Replace with the desired local directory

# Function to ask a question and get the user's response
def ask_question_and_log(prompt, user_response, file_path):
    # Ask the question
    result = pipe(f"<s>[Prompt] {prompt} \n [User response] {user_response} \n [/Response]")

    # Get the generated text (question)
    generated_text = result[0]['generated_text']

    # Print and save the generated text (question)
    print(generated_text)
    with open(file_path, "a", encoding="utf-8") as output_file:
        output_file.write(f"[Model] {generated_text}\n\n")

    # Simulate user answering the question
    user_response = input("Your response: ")  # User provides input

    return user_response

# Initial user response
user_response = "I have chest pain."

# Ask a question based on the user's response and log the conversation
with open(output_file_path, "a", encoding="utf-8") as output_file:
    output_file.write("\nConversation started.\n\n")

# Loop to continue the conversation
while True:
    user_response = ask_question_and_log(prompt, user_response, output_file_path)

    # Check for an exit condition (e.g., user response indicating the end of the conversation)
    if "exit" in user_response.lower():
        print("Ending the conversation.")

        # Save the conversation to a file
        with open(output_file_path, "a", encoding="utf-8") as output_file:
            output_file.write("\nConversation ended by user. ---------------- \n\n")

        break
