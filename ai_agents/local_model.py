import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import spacy
from spacy.training import Example
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
from pathLib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbedding:
    """Custom legal embedding class for handling legal text data.
    """

    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 300):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word2idx = {}
        self.idx2word = {}
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def build_vocab(self, legal_texts: List[str]):
        """Build vocabulary from legal texts."""
        word_freq = {}
        for text in legal_texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_word = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
         self.word2idx = {
            '<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
            '<LEGAL>': 4, '<EVIDENCE>': 5, '<SECTION>': 6
        }

        for i, (word, _) in enumerate(sorted_word[:self.vocab_size - 7], start=7):
            self.word2idx[word] = i + 7
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Convert text to tensor of indices."""
        words = text.lower().split()
        indices = [self.word2idx.get(word, 1) for word in words]

        if len(indices) < max_length:
            indices.extend([0] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        return torch.tensor(indices, dtype=torch.long)


class LegalAttention(nn.Module):
    """Legal document attention mechanism."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        scorces = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_size)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        context=torch.matmul(attention, v)
        context=context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.output(context)

class LegalReasoningTransformer(nn.Module):
    """Transformer model for legal reasoning and argument generation"""

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 1024,
        dropout: float = 0.1
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            LegalTransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_legal_analysis: bool = False
    ):
        batch_size, seq_length = input_ids.size()
        positions = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        x = self.dropout(token_embeddings + position_embeddings)

        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, attention_mask)
            attention_weights.append(attn)

        x = self.layer_norm(x)
        logits = self.output_projection(x)

        output = {
            'logits': logits,
            'attention_weights': attention_weights
        }

        if output_legal_analysis:
            pooled = x.mean(dim=1)
            output.update({
                'legal_section': self.legal_section_analysis(pooled),
                'evidence_relevance': torch.sigmoid(self.evidence_relevance_head(pooled)),
                'argument_strength': self.argument_strength_head(pooled)
            })
        return output

class LegalTransformerBlock(nn.Module):
    """Single transformer block with legal domain adaptation"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = LegalAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size* 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x, attention_weights

class EvidenceAnalysisModel(nn.Module):
    """Specialized model for analyzing evidence in legal documents."""

    def __init__(self, hidden_size: int = 512):
        supper().__init__()
        self.evidence_encoder = nn.Sqequential(
            nn.Linear(300, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
        )

        self.case_encoder = nn.Sequential(
            nn.Linear(300, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
        )

        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.evidence_type_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, evidence_embedding: torch.Tensor, case_embedding: torch.Tensor):
        evidence_repr = self.evidence_encoder(evidence_embedding)
        case_repr = self.case_encoder(case_embedding)

        combined = torch.cat((evidence_repr, case_repr), dim=-1)
        relevance_score = self.relevance_scorer(combined)

        evidence_type = self.evidence_type_classifier(evidence_repr)

        return {
            'relevance_score': relevance_score,
            'evidence_type': evidence_type,
            'evidence_representation': evidence_repr,
        }

class LegalKnowledgeRetriever:
    """Vector-based knowledge retriever for legal documents/knowledge."""

    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.ipc_embeddings = {}
        self.mva_embeddings = {}
        self.crcp_embeddings = {}
        self.cpc_embeddings = {}
        self.case_embeddings = {}
        self.constitution_embeddings = {}

    def build_knowledge_base(self, ipc_data: Dict, mva_data: Dict, crcp_data: Dict, cpc_data: Dict,  case_data: List[Dict], constitution_data: Dict, embedding_model):
        """Build vector representations for legal knowledge."""
        # Process IPC data
        for section, data in ipc_data.items():
            text = f'{data["title"]} {data["description"]}'
            embedding = embedding_model.encode_text(text)
            self.ipc_embeddings[section] = embedding
        # Process MVA data
        for section, data in mva_data.items():
            text = f'{data["title"]} {data["description"]}'
            embedding = embedding_model.encode_text(text)
            self.mva_embeddings[section] = embedding
        # Process CRCP data
        for section, data in crcp_data.items():
            text = f'{data["title"]} {data["description"]}'
            embedding = embedding_model.encode_text(text)
            self.crcp_embeddings[section] = embedding
        # Process CPC data
        for section, data in cpc_data.items():
            text = f'{data["title"]} {data["description"]}'
            embedding = embedding_model.encode_text(text)
            self.cpc_embeddings[section] = embedding

        # Process case data
        for case in case_data:
            case_id = case['id']
            text = f'{case["title"]} {case["summary"]} {case["judgment"]}'
            embedding = embedding_model.encode_text(text)
            self.case_embeddings[case_id] = embedding

        # Process constitution data
        for section, data in constitution_data.items():
            text = f'{data["title"]} {data["content"]}'
            embedding = embedding_model.encode_text(text)
            self.constitution_embeddings[section] = embedding
        logger.info("Knowledge base built successfully with embeddings.")
    
    def retrieve_relevant_sections(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        knowledge_type: str = 'ipc'
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant sections based on query embedding."""
        if knowledge_type == 'ipc':
            embeddings = self.ipc_embeddings
        elif knowledge_type == 'mva':
            embeddings = self.mva_embeddings
        elif knowledge_type == 'crcp':
            embeddings = self.crcp_embeddings
        elif knowledge_type == 'cpc':
            embeddings = self.cpc_embeddings
        elif knowledge_type == 'case':
            embeddings = self.case_embeddings
        elif knowledge_type == 'constitution':
            embeddings = self.constitution_embeddings
        else:
            raise ValueError("Invalid knowledge type specified.")

        similarities = []
        for section_id, section_embedding in embeddings.items():
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                section_embedding.unsqueeze(0)
            ).item()
            similarities.append((section_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class LegalAgentModel(nn.Module):
    """Specialized model for legal agents e.g. Prosecution, Defense, Judge."""

    def __init__(
        self,
        agent_type: str,
        base_model: LegalReasoningTransformer,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.agent_type = agent_type
        self.base_model = base_model

        self.role_embedding = nn.Embedding(3, hidden_size)  # Prosecution, Defense, Judge
        self.strategy_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        if agent_type == 'prosecution':
            self.charge_prediction = nn.Linear(hidden_size, 573)  # Predict charges
            self.evidence_selector = nn.Linear(hidden_size, 1)  # Predict evidence strength

        elif agent_type == 'defense':
            self.charge_prediction = nn.Linear(hidden_size, 10)  # Predict charges
            self.evidence_selector = nn.Linear(hidden_size, hidden_size)  # Predict evidence strength

        elif agent_type == 'judge':
            self.charge_prediction = nn.Linear(hidden_size, 3)  # Predict charges
            self.evidence_selector = nn.Linear(hidden_size, 100)  # Predict evidence strength

        else:
            raise ValueError("Invalid agent type specified.")

    def forward(
        self,
        input_ids: torch.Tensor,
        agent_id: int,
        context_embedding: Optional[torch.Tensor] = None):
        
        base_output = self.base_model(input_ids, output_legal_analysis=True)

        role_embed = self.role_embedding(torch.tensor([agent_id]))

        hidden = base_output['Logits'].mean(dim=1)
        if context_embedding is not None:
            hidden = hidden + context_embedding
        
        strategy = self.strategy_generator(hidden + role_embed)
        output = {
            'base_logits': base_output['logits'],
            'strategy': strategy,
            'legal_analysis': {
                'section': base_output.get('legal_section'),
                'evidence_relevance': base_output.get('evidence_relevance'),
                'argument_strength': base_output.get('argument_strength')
            }
        }

        if self.agent_type == 'prosecution':
            output.update({
                'charge': self.charge_prediction(strategy),
                'evidence_importance': self.evidence_selector(strategy)
            })

        elif self.agent_type == 'defense':
            output.update({
                'defense_type': self.defense_strategy(strategy),
                'counter_arg': self.counter_argument(strategy)
            })

        elif self.agent_type == 'judge':
            output.update({
                'verdict': self.verdict_head(strategy),
                'sentence': self.sentence_predictor(strategy)
            })

        return output

class LegalDataset(Dataset):
    """Dataset for legal documents and cases."""

    def __init__(
        self,
        cases: List[Dict],
        embeddings: LegalEmbedding,
        max_length: int = 512
    ):
        self.cases = cases
        self.embeddings = embeddings
        self.max_length = max_length

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]

        input_text = f"{case['title']} {case['summary']} {case['evidence', '']}"
        input_ids = self.embeddings.encode_text(input_text, self.max_length)

        attention_mask = (input_ids != 0).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'case_id': case['id'],
            'labels': case.get('labels'),
            'metadata': case.get('metadata', {})
        }

class LegalNLPPipeline:
    """spaCy NLP pipeline for legal text/document processing."""

    def __init__(self):
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found, downloading en_core_web_sm...")
            self.nlp = spacy.blank("en")

        self._add_legal_components()

    def _add_legal_components(self):
        """Add legal-specific NLP components to spaCy pipeline."""
        if "legal_ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", name="legal_ner")

            legal_labels = [
                "IPC_SECTION", "MVA_SECTION", "CRCP_SECTION", "CPC_SECTION",
                "CASE_ID", "CONSTITUTION_SECTION", "EVIDENCE_TYPE"
            ]

            for label in legal_labels:
                ner.add_label(label)
        
        @spacy.Language.component("legal_section_extractor")
        def legal_section_extractor(doc):
            import re
            section_pattern = r"(?:Section|Sec\.?)\s*(\d+(?:\w*)?)"

            for match in re.finditer(section_pattern, doc.text, re.IGNORECASE):
                start, end = match.span()
                span = doc.char_span(start, end, label="LEGAL_SECTION")
                if span :
                    doc.ents = list(doc.ents) + [span]
            return doc

            if "legal_section_extractor" not in self.nlp.pipe_names:
                self.nlp.add_pipe("legal_section_extractor", after="ner")
            logger.info("Legal NLP components added to spaCy pipeline.")

        def process_legal_text(self, text: str) -> Dict[str, Any]:
            """Process legal text and extract relevant information"""
            doc = self.nlp(text)

            return {
                'tokens': [token.text for token in doc],
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'sentences': [sent.text for sent in doc.sents],
                'legal_sections': [
                    ent.text for ent in doc.ents if ent.label_ == "LEGAL_SECTION"
                ],
                'key_phrases': self._extract_key_phrases(doc),
                'sentiment': self._analyze_legal_sentiment(doc)
            }
        
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases from legal text."""
        positive_words = ["justice", "fair", "innocent", "lawful", "valid"]
        negative_words = ["guilty", "unlawful", "violated", "breach", "criminal"]
        
        pos_count = sum(1 for token in doc if token.text.lower() in positive_words)
        neg_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return {"positive": 0.5, "negative": 0.5, "neutral": 1.0}
        
        return {
            "positive": pos_count / total,
            "negative": neg_count / total,
            "neutral": 1 - (pos_count + neg_count) / len(doc)
        }

class LegalModelTrainer:
    """Training utility for legal models."""

    def __init__(
        self,
        model: nn.Module
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        self.scheduler = None
        # self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train dor one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = output['logits']
            labels = input_ids.clone()
            labels[:,:-1] = input_ids[:, 1:]
            
            loss = F.cross_entropy(
                logits[:,:-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
                ignore_index=0
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            
        return total_loss / len(dataloader)
    
    def save_model(self, path: str):
        """Save model checkpoint."""

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")


    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":

    embedding = LegalEmbedding()
    model = LegalReasoningTransformer()
    nlp_pipeline = LegalNLPPipeline()

    legal_texts = """
    The accused is charged under Section 302 of the Indian Penal Code for murder.
    The evidence shows that the defendant was present at the crime scene.
    The prosecution argues that the intent to commit murder is clearly established.
    """
    processed = nlp_pipeline.process_legal_text(legal_text)
    print("Processed legal text:")
    print(f"Entities: {processed['entities']}")
    print(f"Legal sections: {processed['legal_sections']}")
    print(f"Key phrases: {processed['key_phrases']}")
    
    # Example model initialization
    prosecutor_model = LegalAgentModel('prosecutor', model)
    defense_model = LegalAgentModel('defense', model)
    judge_model = LegalAgentModel('judge', model)
    
    print("Legal AI models initialized successfully!")




    

    