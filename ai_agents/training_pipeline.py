import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from collections import defaultdict # Added for label mapping
from local_model import (
    LegalEmbeddings, LegalReasoningTransformer, 
    LegalDataset, LegalModelTrainer, LegalNLPPipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_maps = {} # To store mappings for section, chapter, type
        logger.info(f"Using device: {self.device}")
        
    def load_jsonl_data(self, file_path):
        """Load JSONL training data"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def prepare_training_data(self):
        """Prepare training data from your JSONL files and create label mappings"""
        ipc_data = self.load_jsonl_data('dataset/ipc/ipc_training.jsonl')
        mva_data = self.load_jsonl_data('dataset/MVA/mva_training.jsonl')
        crpc_data = self.load_jsonl_data('dataset/crcp/crpc_training.jsonl')
        cpc_data = self.load_jsonl_data('dataset/cpc/cpc_training.jsonl')
        # cons_data = self.load_jsonl_data('dataset/constitution2/constitution_training.jsonl') # Uncommented this line

        all_data = ipc_data + mva_data + crpc_data + cpc_data  # Added cons_data

        training_cases = []
        
        # Collect all unique labels for mapping
        unique_sections = set()
        unique_chapters = set()
        unique_types = set()

        for item in all_data:
            # Ensure section_number, chapter, and type are strings before using them
            # And only add to sets if they are not None
            section_number = str(item['section_number']) if 'section_number' in item and item['section_number'] is not None else None
            chapter = str(item['chapter']) if 'chapter' in item and item['chapter'] is not None else None
            item_type = str(item['type']) if 'type' in item and item['type'] is not None else None

            case = {
                'id': f"{section_number if section_number else 'unknown'}_{item_type if item_type else 'general'}",
                'summary': item['question'],
                'evidence': item['context'],
                'labels': {
                    'answer': item['answer'],
                    'section': section_number,
                    'chapter': chapter,
                    'type': item_type
                },
                'metadata': {
                    'keyword': item.get('keyword', ''),
                    'chapter_title': item.get('chapter_title', '')
                }
            }
            training_cases.append(case)

            # Populate unique sets for label mapping - ONLY add if not None
            if section_number is not None:
                unique_sections.add(section_number)
            if chapter is not None:
                unique_chapters.add(chapter)
            if item_type is not None:
                unique_types.add(item_type)
        
        # Create numerical mappings for labels
        self.label_maps['section_to_id'] = {label: i for i, label in enumerate(sorted(list(unique_sections)))}
        self.label_maps['chapter_to_id'] = {label: i for i, label in enumerate(sorted(list(unique_chapters)))}
        self.label_maps['type_to_id'] = {label: i for i, label in enumerate(sorted(list(unique_types)))}

        # Add reverse mappings for convenience (optional, but good for debugging/inference)
        self.label_maps['id_to_section'] = {i: label for label, i in self.label_maps['section_to_id'].items()}
        self.label_maps['id_to_chapter'] = {i: label for label, i in self.label_maps['chapter_to_id'].items()}
        self.label_maps['id_to_type'] = {i: label for label, i in self.label_maps['type_to_id'].items()}

        logger.info(f"Created label mappings: {len(self.label_maps['section_to_id'])} sections, "
                    f"{len(self.label_maps['chapter_to_id'])} chapters, "
                    f"{len(self.label_maps['type_to_id'])} types.")
        
        return training_cases
    
    def build_embeddings(self, training_cases):
        """Build custom legal embeddings"""
        legal_texts = []
        for case in training_cases:
            legal_texts.append(case['summary'])
            legal_texts.append(case['evidence'])
            legal_texts.append(case['labels']['answer'])
        
        embeddings = LegalEmbeddings(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim']
        )
        
        embeddings.build_vocab(legal_texts)
        logger.info(f"Built vocabulary with {len(embeddings.word2idx)} tokens")
        
        return embeddings
    
    def create_model(self, embeddings):
        """Create the legal reasoning model"""
        model = LegalReasoningTransformer(
            vocab_size=len(embeddings.word2idx),
            hidden_size=self.config['hidden_size'],         # e.g. 512
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            max_seq_length=self.config['max_seq_length'],
            dropout=self.config['dropout']
        )

        model.token_embedding = embeddings.embeddings

        # Fix: Also replace position embedding to match token embedding dim (e.g. 300)
        model.position_embedding = nn.Embedding(
            self.config['max_seq_length'],
            self.config['embedding_dim']                     # e.g. 300
        )

        return model

    def _collate_batch(self, batch):
        """
        Custom collate function for DataLoader to process raw data into tensors.
        This function ensures all inputs and labels are numerical and padded.
        """
        summaries_encoded = []
        evidences_encoded = []
        answers_encoded = []
        section_labels = []
        chapter_labels = []
        type_labels = []

        for item in batch:
            # Encode text fields
            summaries_encoded.append(self.embeddings.encode_text(item['summary'], self.config['max_seq_length']))
            evidences_encoded.append(self.embeddings.encode_text(item['evidence'], self.config['max_seq_length']))
            answers_encoded.append(self.embeddings.encode_text(item['labels']['answer'], self.config['max_seq_length']))

            # Convert string labels to numerical IDs using the mappings
            # Use .get() with a default of -1 if the label is not found in the map
            # This handles cases where a label might be None or an unexpected string
            section_label_id = self.label_maps['section_to_id'].get(item['labels'].get('section'), -1) 
            chapter_label_id = self.label_maps['chapter_to_id'].get(item['labels'].get('chapter'), -1) 
            type_label_id = self.label_maps['type_to_id'].get(item['labels'].get('type'), -1) 

            section_labels.append(section_label_id)
            chapter_labels.append(chapter_label_id)
            type_labels.append(type_label_id)

        # Stack input_ids and attention_masks
        summary_input_ids = torch.stack([s['input_ids'] for s in summaries_encoded])
        summary_attention_mask = torch.stack([s['attention_mask'] for s in summaries_encoded])
        evidence_input_ids = torch.stack([e['input_ids'] for e in evidences_encoded])
        evidence_attention_mask = torch.stack([e['attention_mask'] for e in evidences_encoded])
        answer_input_ids = torch.stack([a['input_ids'] for a in answers_encoded])
        answer_attention_mask = torch.stack([a['attention_mask'] for a in answers_encoded])

        # Convert labels to tensors
        section_labels_tensor = torch.tensor(section_labels, dtype=torch.long)
        chapter_labels_tensor = torch.tensor(chapter_labels, dtype=torch.long)
        type_labels_tensor = torch.tensor(type_labels, dtype=torch.long)

        return {
            'summary_input_ids': summary_input_ids,
            'summary_attention_mask': summary_attention_mask,
            'evidence_input_ids': evidence_input_ids,
            'evidence_attention_mask': evidence_attention_mask,
            'answer_input_ids': answer_input_ids,
            'answer_attention_mask': answer_attention_mask,
            'section_label': section_labels_tensor,
            'chapter_label': chapter_labels_tensor,
            'type_label': type_labels_tensor,
        }
    
    def train_model(self):
        """Main training loop"""
        logger.info("Preparing training data...")
        training_cases = self.prepare_training_data()
        
        logger.info("Building embeddings...")
        embeddings = self.build_embeddings(training_cases)
        # Store embeddings on the pipeline instance so _collate_batch can access it
        self.embeddings = embeddings 
        
        logger.info("Creating model...")
        model = self.create_model(embeddings)
        
        # Create dataset and dataloader
        # Note: LegalDataset might need to be adjusted if it was previously doing tokenization
        # within its __getitem__. With this collate_fn, LegalDataset can simply return the raw dicts.
        dataset = LegalDataset(
            training_cases, 
            embeddings=embeddings, # Keep embeddings for potential internal use or just pass raw data
            max_length=self.config['max_seq_length']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_batch # Use the custom collate function
        )
        
        logger.info("Initializing trainer...")
        trainer = LegalModelTrainer(model, device=str(self.device))
        
        logger.info("Starting training...")
        for epoch in range(self.config['num_epochs']):
            avg_loss = trainer.train_epoch(dataloader, epoch)
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}, Average Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                trainer.save_model(f"checkpoints/legal_model_epoch_{epoch+1}.pt")
        
        trainer.save_model("models/legal_reasoning_model_final.pt")
        torch.save(embeddings, "models/legal_embeddings.pt")
        
        logger.info("Training completed!")
        return model, embeddings

# Training configuration
config = {
    'vocab_size': 50000,
    'embedding_dim': 300,
    'hidden_size': 512,
    'num_layers': 6,
    'num_heads': 8,
    'max_seq_length': 512,
    'dropout': 0.1,
    'batch_size': 8,  # Adjust based on GPU memory
    'num_epochs': 20,
    'learning_rate': 2e-5
}

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    pipeline = LegalTrainingPipeline(config)
    model, embeddings = pipeline.train_model()
    
    print("Legal model training completed successfully!")
    
    # Example inference
    nlp_pipeline = LegalNLPPipeline()
    test_query = "What is Section 302 of IPC about?"
    
    # Encode query
    query_encoded = embeddings.encode_text(test_query).unsqueeze(0)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        # The model's forward pass might need to be updated to accept the new collated batch structure
        # For this example, we're using a single query, so direct encoding is fine.
        # If the model expects specific input keys, adjust this part or the model's forward method.
        outputs = model(query_encoded, output_legal_analysis=True)
        
    print(f"Query: {test_query}")
    print(f"Model output shape: {outputs['logits'].shape}")
    print(f"Legal sections predicted: {outputs['legal_sections'].shape}")
    print(f"Legal chapters predicted: {outputs['legal_chapters'].shape}")
