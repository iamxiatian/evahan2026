from transformers import AutoModelForVision2Seq
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import AutoProcessor

base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
lora_model_path = "output/qwen2_5-vl-7b-evahan-lora-b-augment-0209/checkpoint-2504"
processor_model_path = "output/qwen2_5-vl-7b-evahan-lora-augment-0206-lora-merged"
merged_model_path = f"{lora_model_path}-lora-merged"
base_model = AutoModelForVision2Seq.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, lora_model_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_path)

tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
tokenizer.save_pretrained(merged_model_path)


processor = AutoProcessor.from_pretrained(processor_model_path)
processor.save_pretrained(merged_model_path)