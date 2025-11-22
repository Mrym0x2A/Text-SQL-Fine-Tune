from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch

def main():
	model_name = "unsloth/Llama-3.2-1B-Instruct"
	dataset_name = "xlangai/spider"
	max_seq_length = 1024 
	output_dir = "./llama3.2-spider-finetuned"
	
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=model_name,
		max_seq_length=max_seq_length,
		load_in_4bit=True,  
		dtype=None,
	)
	
	tokenizer = get_chat_template(
		tokenizer,
		chat_template="llama-3.1",
	)
	
	print("Preparing model for QLoRA...")
	model = FastLanguageModel.get_peft_model(
		model,
		r=16,  # LoRA rank
		target_modules=[
			"q_proj", "k_proj", "v_proj", "o_proj",
			"gate_proj", "up_proj", "down_proj",
		],
		lora_alpha=16,
		lora_dropout=0,
		bias="none",
		use_gradient_checkpointing="unsloth",  
		random_state=3407,
	)
	
	print("Loading and preparing dataset...")
	
	def format_dataset(entries):
		conversations = []
		for question, query in zip(entries["question"], entries["query"]):
			conversation = [
	{"role": "user", "content": question},
	{"role": "assistant", "content": query}
			]
			conversations.append(conversation)
		
		texts = [
			tokenizer.apply_chat_template(
	convo, 
	tokenize=False, 
	add_generation_prompt=False
			) for convo in conversations
		]
		return {"text": texts}
	
	dataset = load_dataset(dataset_name, split="train")
	dataset = dataset.map(format_dataset, batched=True)
	
	training_args = TrainingArguments(
		output_dir=output_dir,
		per_device_train_batch_size=1,  
		gradient_accumulation_steps=8,   
		warmup_steps=10,
		max_steps=100,	   
		learning_rate=2e-4,
		fp16=not torch.cuda.is_bf16_supported(),
		bf16=torch.cuda.is_bf16_supported(),
		logging_steps=10,
		optim="adamw_8bit",
		weight_decay=0.01,
		lr_scheduler_type="linear",
		seed=3407,
		report_to="none",
		save_steps=50,
		save_total_limit=2,
		dataloader_pin_memory=False,	 
	)
	
	print("Setting up trainer...")
	trainer = SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		train_dataset=dataset,
		dataset_text_field="text",
		max_seq_length=max_seq_length,
		args=training_args,
		packing=False,  
	)
	
	print("Starting training...")
	trainer.train()
	
	print("Saving model...")
	trainer.save_model()
	tokenizer.save_pretrained(output_dir)
	
	print(f"Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
	main()
