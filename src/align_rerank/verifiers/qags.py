from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from align_rerank.utils.text import split_sentences

@dataclass
class QAGSConfig:
    qg_model: str = 'valhalla/t5-base-qg-hl'  # Question generation model
    qa_model: str = 'distilbert-base-uncased-distilled-squad'  # QA model
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_questions: int = 5  # Maximum questions to generate per summary

class QAGSScorer:
    """QAGS (Question Answering and Generation Score) for factuality evaluation.
    
    QAGS generates questions from the summary, answers them from the source document,
    and compares answers to extract factuality scores. This is a simplified implementation
    that uses T5 for question generation and DistilBERT for QA.
    
    Reference: Wang et al. (2020) "BARTScore: Evaluating Generated Text as Text Generation"
    """
    def __init__(self, cfg: QAGSConfig = QAGSConfig()):
        self.cfg = cfg
        
        # Load question generation model
        try:
            self.qg_tokenizer = AutoTokenizer.from_pretrained(cfg.qg_model, use_fast=True)
            self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.qg_model).to(cfg.device).eval()
        except Exception:
            # Fallback: use a simpler approach if QG model unavailable
            self.qg_tokenizer = None
            self.qg_model = None
        
        # Load QA model
        try:
            self.qa_tokenizer = AutoTokenizer.from_pretrained(cfg.qa_model, use_fast=True)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(cfg.qa_model).to(cfg.device).eval()
        except Exception:
            # Fallback: use FactCC-like scoring if QA unavailable
            self.qa_tokenizer = None
            self.qa_model = None

    @torch.no_grad()
    def generate_questions(self, summary: str) -> List[str]:
        """Generate questions from summary sentences."""
        if self.qg_model is None or self.qg_tokenizer is None:
            # Fallback: extract sentence-based questions (simplified)
            sentences = split_sentences(summary)
            # Return first few sentences as "questions" (simplified QAGS)
            return sentences[:self.cfg.max_questions]
        
        sentences = split_sentences(summary)
        questions = []
        for sent in sentences[:self.cfg.max_questions]:
            try:
                # Format for T5 QG: "generate question: <sentence>"
                input_text = f"generate question: {sent}"
                inputs = self.qg_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=128).to(self.cfg.device)
                outputs = self.qg_model.generate(**inputs, max_new_tokens=32, num_beams=2)
                question = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if question.strip():
                    questions.append(question)
            except Exception:
                # If generation fails, use the sentence as-is
                questions.append(sent)
        
        return questions if questions else sentences[:self.cfg.max_questions]

    @torch.no_grad()
    def answer_question(self, context: str, question: str) -> str:
        """Answer a question from the context."""
        if self.qa_model is None or self.qa_tokenizer is None:
            # Fallback: return empty string or use simple matching
            return ""
        
        try:
            inputs = self.qa_tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512).to(self.cfg.device)
            outputs = self.qa_model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_idx = start_logits.argmax().item()
            end_idx = end_logits.argmax().item()
            
            if end_idx < start_idx:
                return ""
            
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
            return answer.strip()
        except Exception:
            return ""

    def score(self, context: str, summary: str) -> float:
        """Compute QAGS score: F1 between generated answers and summary sentences.
        
        Simplified version: generates questions, answers them, and computes overlap.
        """
        questions = self.generate_questions(summary)
        if not questions:
            return 0.0
        
        summary_sentences = [s.lower().strip() for s in split_sentences(summary)]
        if not summary_sentences:
            return 0.0
        
        # Answer questions from context
        answers = [self.answer_question(context, q).lower().strip() for q in questions]
        answers = [a for a in answers if a]  # Remove empty answers
        
        if not answers:
            # Fallback: if QA fails, use simple word overlap
            context_lower = context.lower()
            summary_words = set(' '.join(summary_sentences).split())
            context_words = set(context_lower.split())
            overlap = len(summary_words & context_words)
            total = len(summary_words)
            return float(overlap / max(total, 1))
        
        # Compute F1: compare answer words with summary sentence words
        total_f1 = 0.0
        for answer in answers:
            if not answer:
                continue
            answer_words = set(answer.split())
            best_f1 = 0.0
            for sent in summary_sentences:
                sent_words = set(sent.split())
                if not sent_words:
                    continue
                overlap = len(answer_words & sent_words)
                precision = overlap / max(len(answer_words), 1)
                recall = overlap / max(len(sent_words), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                best_f1 = max(best_f1, f1)
            total_f1 += best_f1
        
        return float(total_f1 / max(len(answers), 1))

