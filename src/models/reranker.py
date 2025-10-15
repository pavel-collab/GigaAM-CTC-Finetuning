import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple

class TransformerLMReranker:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def score_sequence(self, text: str) -> float:
        """Вычисляет log-вероятность текста с помощью трансформерной LM"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # Возвращаем отрицательный loss (чем выше, тем лучше)
            return -outputs.loss.item()
    
    def rerank_nbest(self, nbest_list: List[Dict], alpha: float = 0.7) -> List[Dict]:
        """Переранжирует n-best список с комбинацией CTC и LM scores"""
        reranked = []
        
        for candidate in nbest_list:
            ctc_score = candidate['score']
            lm_score = self.score_sequence(candidate['hyp'])
            
            # Комбинируем scores: alpha * CTC + (1-alpha) * LM
            combined_score = alpha * ctc_score + (1 - alpha) * lm_score
            reranked.append({
                'hyp': candidate['hyp'],
                'ctc_score': ctc_score,
                'lm_score': lm_score,
                'combined_score': combined_score
            })
        
        # Сортируем по комбинированному score
        return sorted(reranked, key=lambda x: x['combined_score'], reverse=True)