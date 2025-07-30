""" Indian Constitution Data Cleaner - Enhanced for CSV Input & RAG
 Comprehensive tool for cleaning and structuring constitutional text data from CSV files
 Optimized for RAG (Retrieval Augmented Generation) systems
"""

import json
import re
import os
import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import argparse
from datetime import datetime
import hashlib

class ConstitutionDataCleaner:
    """Main class for cleaning and processing constitutional text data from CSV files."""
    
    def __init__(self):
        self.cleaned_articles = []
        self.qa_pairs = []
        self.rag_chunks = []
        self.statistics = {}
        
        # Common words to exclude from keyword extraction
        self.stop_words = {
            'the', 'and', 'or', 'of', 'in', 'to', 'a', 'an', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'such', 'any',
            'all', 'no', 'not', 'by', 'for', 'with', 'as', 'on', 'at', 'from', 'under',
            'over', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their',
            'he', 'she', 'it', 'they', 'him', 'them', 'who', 'which', 'what', 'where',
            'when', 'why', 'how', 'if', 'unless', 'until', 'while', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'article', 'constitution', 'indian',
            'union', 'state', 'states', 'parliament', 'government', 'act', 'law'
        }
        
        # Constitutional themes for better categorization
        self.constitutional_themes = {
            'fundamental_rights': ['right', 'freedom', 'liberty', 'equality', 'protection', 'discrimination'],
            'directive_principles': ['directive', 'principle', 'policy', 'welfare', 'social', 'economic'],
            'federal_structure': ['union', 'state', 'territory', 'federal', 'jurisdiction', 'distribution'],
            'governance': ['executive', 'legislative', 'judicial', 'president', 'minister', 'governor'],
            'amendment': ['amendment', 'modify', 'alter', 'change', 'repeal'],
            'emergency': ['emergency', 'proclamation', 'national', 'financial', 'constitutional']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common formatting issues
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('…', '...')
        text = text.replace('Rep by', 'Replaced by')
        text = text.replace('w e f', 'with effect from')
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s*', r'\1 ', text)
        
        # Fix parentheses spacing
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract important keywords from text."""
        # Convert to lowercase and remove punctuation
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Filter out stop words and short words
        filtered_words = [
            word for word in words 
            if len(word) > 3 and word not in self.stop_words
        ]
        
        # Count word frequency
        word_counts = Counter(filtered_words)
        
        # Return most common words
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def identify_theme(self, text: str) -> List[str]:
        """Identify constitutional themes in the text."""
        text_lower = text.lower()
        identified_themes = []
        
        for theme, keywords in self.constitutional_themes.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_themes.append(theme)
        
        return identified_themes
    
    def parse_article_number(self, text: str) -> Optional[str]:
        """Extract article number from text."""
        # Pattern for article numbers at the beginning (including 2A, 12A, etc.)
        article_pattern = r'^"?(\d+[A-Z]?)\.\s+'
        match = re.match(article_pattern, text.strip())
        return match.group(1) if match else None
    
    def extract_article_title(self, text: str) -> str:
        """Extract article title from text."""
        # Remove article number and quotes
        title = re.sub(r'^"?\d+[A-Z]?\.\s+', '', text.strip())
        
        # Find the end of title (usually before first parenthesis or colon)
        patterns = [
            r'^([^:(]+?)(?:\s*:|\s*\()',  # Before colon or parenthesis
            r'^([^.]+?)(?:\s+\([0-9])',   # Before numbered clause
            r'^(.{1,100}?)(?:\s+[A-Z][a-z]+\s+may\s+)',  # Before "Parliament may" etc.
            r'^(.{1,150})'  # Fallback: first 150 chars
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                extracted_title = match.group(1).strip()
                if len(extracted_title) > 5:  # Ensure meaningful title
                    return self.clean_text(extracted_title)
        
        return self.clean_text(title[:100] + '...' if len(title) > 100 else title)
    
    def parse_clauses(self, text: str) -> List[Dict[str, str]]:
        """Parse clauses from article text."""
        clauses = []
        
        # Remove article number and title first
        content = re.sub(r'^"?\d+[A-Z]?\.\s+[^:(]+?(?:\s*:|\s*\()', '', text.strip())
        
        # Patterns for different clause types
        clause_patterns = [
            (r'\((\d+)\)\s+([^()]+?)(?=\s*\(\d+\)|$)', 'numbered'),  # (1), (2), etc.
            (r'\(([a-z])\)\s+([^()]+?)(?=\s*\([a-z]\)|$)', 'lettered'),  # (a), (b), etc.
            (r'\(([ivxlc]+)\)\s+([^()]+?)(?=\s*\([ivxlc]+\)|$)', 'roman'),  # (i), (ii), etc.
        ]
        
        for pattern, clause_type in clause_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clause_id = f"({match.group(1)})"
                clause_text = self.clean_text(match.group(2))
                if len(clause_text) > 10:  # Only meaningful clauses
                    clauses.append({
                        'clause_id': clause_id,
                        'text': clause_text,
                        'type': clause_type
                    })
        
        return clauses
    
    def identify_amendments(self, text: str) -> List[str]:
        """Extract amendment information from text."""
        amendments = []
        
        # Patterns for amendments
        amendment_patterns = [
            r'(Constitution[^,]+Amendment Act[^,]+(?:,\s*\d{4})?)',
            r'(Replaced by[^,]+Amendment Act[^,]+(?:,\s*\d{4})?)',
            r'(Inserted by[^,]+Amendment Act[^,]+(?:,\s*\d{4})?)',
            r'(Substituted by[^,]+Amendment Act[^,]+(?:,\s*\d{4})?)'
        ]
        
        for pattern in amendment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amendment = self.clean_text(match.group(1))
                if amendment not in amendments:
                    amendments.append(amendment)
        
        return amendments
    
    def read_csv_file(self, csv_file: str) -> List[str]:
        """Read CSV file and extract Article column."""
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Check if 'Article' column exists
            if 'Article' not in df.columns:
                print(f"Warning: 'Article' column not found in CSV. Available columns: {list(df.columns)}")
                # Try to find similar column names
                for col in df.columns:
                    if 'article' in col.lower():
                        print(f"Using column '{col}' instead of 'Article'")
                        return df[col].dropna().tolist()
                raise ValueError("No suitable article column found")
            
            # Extract articles and remove empty/null values
            articles = df['Article'].dropna().tolist()
            print(f"Successfully read {len(articles)} articles from CSV")
            return articles
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []
    
    def parse_constitution_from_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """Parse constitutional text from CSV file."""
        # Read CSV data
        article_texts = self.read_csv_file(csv_file)
        
        if not article_texts:
            print("No article data found in CSV file")
            return []
        
        articles = []
        current_section = "Unknown"
        
        for i, article_text in enumerate(article_texts):
            if not article_text or len(article_text.strip()) < 10:
                continue
            
            # Extract article number
            article_num = self.parse_article_number(article_text)
            
            if not article_num:
                # If no article number found, use index
                article_num = str(i + 1)
            
            # Extract title
            title = self.extract_article_title(article_text)
            
            # Parse clauses
            clauses = self.parse_clauses(article_text)
            
            # Extract amendments
            amendments = self.identify_amendments(article_text)
            
            # Clean content
            clean_content = self.clean_text(article_text)
            
            # Extract keywords
            keywords = self.extract_keywords(clean_content)
            
            # Identify themes
            themes = self.identify_theme(clean_content)
            
            # Create article object
            article = {
                'article_number': article_num,
                'title': title,
                'content': clean_content,
                'section': current_section,
                'clauses': clauses,
                'keywords': keywords,
                'themes': themes,
                'amendments': amendments,
                'word_count': len(clean_content.split()),
                'full_text': f"Article {article_num}: {title}\n\n{clean_content}",
                'id': f"article_{article_num}",
                'hash': hashlib.md5(clean_content.encode()).hexdigest()[:12]
            }
            
            articles.append(article)
        
        return articles
    
    def create_rag_chunks(self, articles: List[Dict[str, Any]], chunk_size: int = 500) -> List[Dict[str, Any]]:
        """Create optimized chunks for RAG systems."""
        chunks = []
        
        for article in articles:
            # Full article chunk
            chunks.append({
                'id': f"chunk_article_{article['article_number']}",
                'text': article['full_text'],
                'type': 'full_article',
                'metadata': {
                    'article_number': article['article_number'],
                    'title': article['title'],
                    'section': article['section'],
                    'keywords': article['keywords'],
                    'themes': article['themes'],
                    'word_count': article['word_count'],
                    'has_clauses': len(article['clauses']) > 0,
                    'has_amendments': len(article['amendments']) > 0
                },
                'embedding_text': f"Article {article['article_number']}: {article['title']}. {article['content'][:300]}...",
                'search_terms': article['keywords'] + article['themes'] + [article['title'].lower()]
            })
            
            # Individual clause chunks
            for clause in article['clauses']:
                if len(clause['text']) > 50:  # Only substantial clauses
                    chunks.append({
                        'id': f"chunk_article_{article['article_number']}_clause_{clause['clause_id']}",
                        'text': f"Article {article['article_number']}, Clause {clause['clause_id']}: {clause['text']}",
                        'type': 'clause',
                        'metadata': {
                            'article_number': article['article_number'],
                            'clause_id': clause['clause_id'],
                            'clause_type': clause['type'],
                            'parent_title': article['title'],
                            'section': article['section'],
                            'themes': article['themes']
                        },
                        'embedding_text': clause['text'],
                        'search_terms': self.extract_keywords(clause['text'], 5)
                    })
            
            # Split long articles into smaller chunks
            if article['word_count'] > chunk_size:
                words = article['content'].split()
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    chunks.append({
                        'id': f"chunk_article_{article['article_number']}_part_{i//chunk_size + 1}",
                        'text': chunk_text,
                        'type': 'partial_article',
                        'metadata': {
                            'article_number': article['article_number'],
                            'part_number': i//chunk_size + 1,
                            'parent_title': article['title'],
                            'section': article['section'],
                            'themes': article['themes']
                        },
                        'embedding_text': chunk_text,
                        'search_terms': self.extract_keywords(chunk_text, 5)
                    })
        
        return chunks
    
    def generate_enhanced_qa_pairs(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate comprehensive QA pairs optimized for RAG systems."""
        qa_pairs = []
        
        for article in articles:
            article_num = article['article_number']
            title = article['title']
            content = article['content']
            
            # Basic overview questions
            qa_pairs.extend([
                {
                    'question': f"What is Article {article_num} of the Indian Constitution?",
                    'answer': f"Article {article_num} is titled '{title}'. {content[:300]}...",
                    'context': article['full_text'],
                    'article_number': article_num,
                    'type': 'overview',
                    'difficulty': 'basic',
                    'themes': article['themes']
                },
                {
                    'question': f"What is the title of Article {article_num}?",
                    'answer': title,
                    'context': article['full_text'],
                    'article_number': article_num,
                    'type': 'title',
                    'difficulty': 'basic',
                    'themes': article['themes']
                },
                {
                    'question': f"Explain Article {article_num} in detail.",
                    'answer': f"Article {article_num} ({title}) states: {content}",
                    'context': article['full_text'],
                    'article_number': article_num,
                    'type': 'detailed_explanation',
                    'difficulty': 'intermediate',
                    'themes': article['themes']
                }
            ])
            
            # Clause-specific questions
            for clause in article['clauses']:
                if len(clause['text']) > 30:
                    qa_pairs.extend([
                        {
                            'question': f"What does clause {clause['clause_id']} of Article {article_num} state?",
                            'answer': clause['text'],
                            'context': article['full_text'],
                            'article_number': article_num,
                            'clause': clause['clause_id'],
                            'type': 'clause_specific',
                            'difficulty': 'intermediate',
                            'themes': article['themes']
                        },
                        {
                            'question': f"Explain clause {clause['clause_id']} of Article {article_num}.",
                            'answer': f"Clause {clause['clause_id']} of Article {article_num} explains: {clause['text']}",
                            'context': article['full_text'],
                            'article_number': article_num,
                            'clause': clause['clause_id'],
                            'type': 'clause_explanation',
                            'difficulty': 'advanced',
                            'themes': article['themes']
                        }
                    ])
            
            # Theme-based questions
            for theme in article['themes']:
                qa_pairs.append({
                    'question': f"Which article deals with {theme.replace('_', ' ')} in the Indian Constitution?",
                    'answer': f"Article {article_num} ({title}) deals with {theme.replace('_', ' ')}.",
                    'context': article['full_text'],
                    'article_number': article_num,
                    'theme': theme,
                    'type': 'theme_based',
                    'difficulty': 'intermediate',
                    'themes': article['themes']
                })
            
            # Keyword-based questions
            for keyword in article['keywords'][:5]:
                qa_pairs.append({
                    'question': f"What does the Constitution say about {keyword}?",
                    'answer': f"According to Article {article_num} ({title}), {content[:200]}...",
                    'context': article['full_text'],
                    'article_number': article_num,
                    'keyword': keyword,
                    'type': 'keyword_search',
                    'difficulty': 'intermediate',
                    'themes': article['themes']
                })
            
            # Amendment questions
            for amendment in article['amendments']:
                qa_pairs.append({
                    'question': f"What amendments have been made to Article {article_num}?",
                    'answer': f"Article {article_num} has been modified by: {amendment}",
                    'context': article['full_text'],
                    'article_number': article_num,
                    'amendment': amendment,
                    'type': 'amendment_based',
                    'difficulty': 'advanced',
                    'themes': article['themes']
                })
        
        return qa_pairs
    
    def calculate_statistics(self, articles: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics."""
        total_words = sum(article['word_count'] for article in articles)
        total_clauses = sum(len(article['clauses']) for article in articles)
        total_keywords = sum(len(article['keywords']) for article in articles)
        
        # Theme distribution
        theme_counts = Counter()
        for article in articles:
            for theme in article['themes']:
                theme_counts[theme] += 1
        
        # QA type distribution
        qa_types = Counter(qa['type'] for qa in qa_pairs)
        
        # Chunk type distribution
        chunk_types = Counter(chunk['type'] for chunk in chunks)
        
        return {
            'total_articles': len(articles),
            'total_words': total_words,
            'total_clauses': total_clauses,
            'total_keywords': total_keywords,
            'total_qa_pairs': len(qa_pairs),
            'total_rag_chunks': len(chunks),
            'avg_words_per_article': total_words / len(articles) if articles else 0,
            'avg_clauses_per_article': total_clauses / len(articles) if articles else 0,
            'theme_distribution': dict(theme_counts),
            'qa_types_distribution': dict(qa_types),
            'chunk_types_distribution': dict(chunk_types),
            'articles_with_amendments': sum(1 for a in articles if a['amendments']),
            'articles_with_clauses': sum(1 for a in articles if a['clauses']),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def export_json(self, data: Any, filename: str) -> None:
        """Export data to JSON format."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Exported: {filename}")
    
    def export_rag_formats(self, chunks: List[Dict[str, Any]]) -> None:
        """Export data in various RAG-optimized formats."""
        
        # Standard RAG format
        rag_data = {
            'documents': chunks,
            'metadata': {
                'total_chunks': len(chunks),
                'chunk_types': list(set(chunk['type'] for chunk in chunks)),
                'created_at': datetime.now().isoformat(),
                'source': 'Indian Constitution CSV'
            }
        }
        self.export_json(rag_data, 'constitution_rag_chunks.json')
        
        # Vector database format (Pinecone, Weaviate, etc.)
        vector_data = []
        for chunk in chunks:
            vector_data.append({
                'id': chunk['id'],
                'text': chunk['embedding_text'],
                'metadata': chunk['metadata']
            })
        self.export_json(vector_data, 'constitution_vector_db.json')
        
        # LangChain Document format
        langchain_docs = []
        for chunk in chunks:
            langchain_docs.append({
                'page_content': chunk['text'],
                'metadata': {
                    **chunk['metadata'],
                    'source': f"Article {chunk['metadata'].get('article_number', 'Unknown')}",
                    'chunk_type': chunk['type']
                }
            })
        self.export_json(langchain_docs, 'constitution_langchain_docs.json')
        
        # Search index format
        search_index = []
        for chunk in chunks:
            search_index.append({
                'id': chunk['id'],
                'title': chunk['metadata'].get('title', f"Article {chunk['metadata'].get('article_number', '')}"),
                'content': chunk['text'],
                'keywords': chunk.get('search_terms', []),
                'category': chunk['metadata'].get('section', 'Constitution'),
                'themes': chunk['metadata'].get('themes', [])
            })
        self.export_json(search_index, 'constitution_search_index.json')
    
    def export_csv_formats(self, articles: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> None:
        """Export data in CSV formats."""
        
        # Enhanced Articles CSV
        articles_df = pd.DataFrame([
            {
                'Article_Number': article['article_number'],
                'Title': article['title'],
                'Content': article['content'][:500] + '...' if len(article['content']) > 500 else article['content'],
                'Section': article['section'],
                'Word_Count': article['word_count'],
                'Clause_Count': len(article['clauses']),
                'Keywords': ', '.join(article['keywords'][:10]),
                'Themes': ', '.join(article['themes']),
                'Has_Amendments': len(article['amendments']) > 0,
                'Amendment_Count': len(article['amendments'])
            }
            for article in articles
        ])
        articles_df.to_csv('constitution_articles_enhanced.csv', index=False, encoding='utf-8')
        print("✓ Exported: constitution_articles_enhanced.csv")
        
        # Enhanced QA Pairs CSV
        qa_df = pd.DataFrame([
            {
                'Question': qa['question'],
                'Answer': qa['answer'][:300] + '...' if len(qa['answer']) > 300 else qa['answer'],
                'Article_Number': qa['article_number'],
                'Type': qa['type'],
                'Difficulty': qa['difficulty'],
                'Themes': ', '.join(qa.get('themes', [])),
                'Keyword': qa.get('keyword', ''),
                'Clause': qa.get('clause', ''),
                'Theme': qa.get('theme', '')
            }
            for qa in qa_pairs
        ])
        qa_df.to_csv('constitution_qa_enhanced.csv', index=False, encoding='utf-8')
        print("✓ Exported: constitution_qa_enhanced.csv")
        
        # RAG Chunks CSV
        chunks_df = pd.DataFrame([
            {
                'Chunk_ID': chunk['id'],
                'Text': chunk['text'][:400] + '...' if len(chunk['text']) > 400 else chunk['text'],
                'Type': chunk['type'],
                'Article_Number': chunk['metadata'].get('article_number', ''),
                'Title': chunk['metadata'].get('title', ''),
                'Section': chunk['metadata'].get('section', ''),
                'Themes': ', '.join(chunk['metadata'].get('themes', [])),
                'Search_Terms': ', '.join(chunk.get('search_terms', []))
            }
            for chunk in chunks
        ])
        chunks_df.to_csv('constitution_rag_chunks.csv', index=False, encoding='utf-8')
        print("✓ Exported: constitution_rag_chunks.csv")
    
    def export_training_formats(self, qa_pairs: List[Dict[str, Any]]) -> None:
        """Export data in training-ready formats."""
        
        # JSONL format for fine-tuning
        with open('constitution_training.jsonl', 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                training_entry = {
                    "messages": [
                        {"role": "system", "content": "You are an expert on the Indian Constitution. Provide accurate, detailed, and contextually relevant information about constitutional articles, clauses, and provisions. Focus on being helpful for legal research and educational purposes."},
                        {"role": "user", "content": qa['question']},
                        {"role": "assistant", "content": qa['answer']}
                    ],
                    "metadata": {
                        "article_number": qa['article_number'],
                        "type": qa['type'],
                        "difficulty": qa['difficulty'],
                        "themes": qa.get('themes', [])
                    }
                }
                f.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
        print("✓ Exported: constitution_training.jsonl")
        
        # Chat completion format
        with open('constitution_chat_format.jsonl', 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                chat_entry = {
                    "instruction": qa['question'],
                    "input": f"Context: {qa.get('context', '')[:200]}...",
                    "output": qa['answer']
                }
                f.write(json.dumps(chat_entry, ensure_ascii=False) + '\n')
        print("✓ Exported: constitution_chat_format.jsonl")
    
    def process_csv_file(self, csv_file: str) -> None:
        """Process a CSV file containing constitutional data."""
        print(f"Processing CSV file: {csv_file}")
        print("=" * 50)
        
        # Parse articles from CSV
        print("📖 Parsing constitutional articles from CSV...")
        self.cleaned_articles = self.parse_constitution_from_csv(csv_file)
        
        if not self.cleaned_articles:
            print("❌ No articles found in CSV file!")
            return
        
        print(f"✅ Parsed {len(self.cleaned_articles)} articles")
        
        # Generate RAG chunks
        print("🔄 Creating RAG-optimized chunks...")
        self.rag_chunks = self.create_rag_chunks(self.cleaned_articles)
        print(f"✅ Created {len(self.rag_chunks)} RAG chunks")
        
        # Generate QA pairs
        print("❓ Generating enhanced QA pairs...")
        self.qa_pairs = self.generate_enhanced_qa_pairs(self.cleaned_articles)
        print(f"✅ Generated {len(self.qa_pairs)} QA pairs")
        
        # Calculate statistics
        print("📊 Calculating statistics...")
        self.statistics = self.calculate_statistics(self.cleaned_articles, self.qa_pairs, self.rag_chunks)
        
        # Export all formats
        print("\n📁 Exporting data in multiple formats...")
        self.export_json(self.cleaned_articles, 'constitution_structured.json')
        self.export_json(self.qa_pairs, 'constitution_qa_pairs.json')
        self.export_json(self.statistics, 'constitution_statistics.json')
        
        self.export_rag_formats(self.rag_chunks)
        self.export_csv_formats(self.cleaned_articles, self.qa_pairs, self.rag_chunks)
        self.export_training_formats(self.qa_pairs)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print comprehensive processing summary."""
        print("\n" + "=" * 70)
        print("🎉 CONSTITUTION DATA PROCESSING COMPLETE!")
        print("=" * 70)
        
        # Basic statistics
        print(f"📊 PROCESSING STATISTICS:")
        print(f"   📄 Total Articles Processed: {self.statistics['total_articles']}")
        print(f"   📝 Total Words: {self.statistics['total_words']:,}")
        print(f"   📋 Total Clauses: {self.statistics['total_clauses']}")
        print(f"   🔍 Total Keywords: {self.statistics['total_keywords']}")
        print(f"   ❓ QA Pairs Generated: {self.statistics['total_qa_pairs']}")
        print(f"   🧩 RAG Chunks Created: {self.statistics['total_rag_chunks']}")
        print(f"   📈 Avg Words/Article: {self.statistics['avg_words_per_article']:.1f}")
        print(f"   ⚖️ Articles with Amendments: {self.statistics['articles_with_amendments']}")
        print(f"   📑 Articles with Clauses: {self.statistics['articles_with_clauses']}")
        
        # Theme distribution
        if self.statistics['theme_distribution']:
            print(f"\n🎯 CONSTITUTIONAL THEMES IDENTIFIED:")
            for theme, count in sorted(self.statistics['theme_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"   • {theme.replace('_', ' ').title()}: {count} articles")
        
        # QA distribution
        if self.statistics['qa_types_distribution']:
            print(f"\n❓ QA PAIRS BY TYPE:")
            for qa_type, count in sorted(self.statistics['qa_types_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"   • {qa_type.replace('_', ' ').title()}: {count} pairs")
        
        # Chunk distribution
        if self.statistics['chunk_types_distribution']:
            print(f"\n🧩 RAG CHUNKS BY TYPE:")
            for chunk_type, count in sorted(self.statistics['chunk_types_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"   • {chunk_type.replace('_', ' ').title()}: {count} chunks")
        
        print(f"\n📁 EXPORTED FILES:")
        files = [
            ("constitution_structured.json", "Structured article data with metadata"),
            ("constitution_qa_pairs.json", "Question-answer pairs for training"),
            ("constitution_statistics.json", "Comprehensive processing statistics"),
            ("constitution_rag_chunks.json", "RAG-optimized chunks with metadata"),
            ("constitution_vector_db.json", "Vector database ready format"),
            ("constitution_langchain_docs.json", "LangChain document format"),
            ("constitution_search_index.json", "Search engine optimized format"),
            ("constitution_articles_enhanced.csv", "Enhanced articles in CSV"),
            ("constitution_qa_enhanced.csv", "Enhanced QA pairs in CSV"),
            ("constitution_rag_chunks.csv", "RAG chunks in CSV format"),
            ("constitution_training.jsonl", "Fine-tuning training data"),
            ("constitution_chat_format.jsonl", "Chat completion training format")
        ]
        
        for filename, description in files:
            print(f"   ✅ {filename} - {description}")
        
        print(f"\n🚀 READY FOR USE WITH:")
        use_cases = [
            "🤖 RAG (Retrieval Augmented Generation) Systems",
            "🔍 Vector Databases (Pinecone, Weaviate, Chroma)",
            "🦜 LangChain and LlamaIndex Applications",
            "🧠 Machine Learning Model Training",
            "📚 Legal Research and Analysis Tools",
            "🎓 Educational Applications and Chatbots",
            "⚖️ Constitutional Knowledge Bases",
            "🔎 Semantic Search Systems",
            "💬 Question-Answering Applications",
            "📖 Legal Document Processing"
        ]
        
        for use_case in use_cases:
            print(f"   {use_case}")
        
        print(f"\n💡 RAG OPTIMIZATION FEATURES:")
        optimization_features = [
            "📏 Multiple chunk sizes for different use cases",
            "🏷️ Rich metadata for enhanced filtering",
            "🎯 Theme-based categorization",
            "🔑 Keyword extraction for better search",
            "📝 Context preservation across chunks",
            "🔗 Article-clause relationship mapping",
            "📊 Statistical analysis for quality control",
            "🎪 Multiple export formats for flexibility"
        ]
        
        for feature in optimization_features:
            print(f"   {feature}")
        
        print(f"\n⏰ Processing completed at: {self.statistics['processing_timestamp']}")
        print("=" * 70)

def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(
        description='Clean and process Indian Constitution CSV data for RAG systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python constitution_cleaner.py constitution.csv
  python constitution_cleaner.py data.csv --output-dir ./processed_data
  
The CSV file should have an 'Article' column containing constitutional text.
        """
    )
    parser.add_argument('input_file', help='Input CSV file containing constitutional articles')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for processed files')
    parser.add_argument('--chunk-size', '-c', type=int, default=500, help='Maximum words per RAG chunk (default: 500)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        return
    
    if not args.input_file.lower().endswith('.csv'):
        print(f"❌ Error: Input file must be a CSV file!")
        return
    
    # Create output directory
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
        print(f"📁 Output directory: {os.path.abspath(args.output_dir)}")
    
    # Process the file
    cleaner = ConstitutionDataCleaner()
    cleaner.process_csv_file(args.input_file)

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(os.sys.argv) == 1:
        print("🏛️  Indian Constitution Data Cleaner & RAG Optimizer")
        print("=" * 55)
        print("Enhanced for CSV input and RAG system integration\n")
        
        # Check for default CSV file
        default_files = ['const.csv', 'constitution.csv', 'articles.csv']
        input_file = None
        
        for file in default_files:
            if os.path.exists(file):
                input_file = file
                print(f"📄 Found default file: {file}")
                break
        
        if not input_file:
            print("❌ No default CSV file found!")
            print("\n📋 Usage:")
            print("   python constitution_cleaner.py <csv_file>")
            print("\n📁 Expected CSV format:")
            print("   - Must have an 'Article' column")
            print("   - Each row should contain one constitutional article")
            print("   - Example: '1. Name and territory of the Union (1) India...'")
            print("\n💡 Place your CSV file as 'const.csv' for auto-detection")
        else:
            print(f"🚀 Processing {input_file}...\n")
            cleaner = ConstitutionDataCleaner()
            cleaner.process_csv_file(input_file)
    else:
        main()