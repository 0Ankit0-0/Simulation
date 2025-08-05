import json
import re
import os
from typing import Dict, List, Any
import pandas as pd

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common formatting issues
    text = text.replace('—', '-')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    return text

def process_constitution_data():
    """Process the Constitution.json file and create cleaned datasets."""
    
    # Check if Constitution.json exists
    if not os.path.exists('constitution.json'):
        print("Error: Constitution.json file not found in current directory!")
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        return
    
    # Load the data
    print("Loading Constitution.json...")
    with open('Constitution.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} articles")
    
    # Process articles
    cleaned_articles = []
    qa_pairs = []
    
    for article in raw_data:
        # Extract article info
        article_num = str(article.get('article', '')).strip('()')
        title = clean_text(article.get('title', ''))
        source = article.get('source', '')
        
        # Process clauses
        clauses = []
        all_keywords = set()
        
        for point in article.get('points', []):
            clause_info = {
                'clause': point.get('clause', ''),
                'content': clean_text(point.get('content', '')),
                'keywords': point.get('keywords', []),
                'notes': point.get('notes', '')
            }
            clauses.append(clause_info)
            all_keywords.update(clause_info['keywords'])
        
        # Create full text
        full_text_parts = [f"Article {article_num}: {title}"]
        for clause in clauses:
            if clause['clause']:
                full_text_parts.append(f"{clause['clause']} {clause['content']}")
            else:
                full_text_parts.append(clause['content'])
        
        full_text = '\\n'.join(full_text_parts)
        
        # Create cleaned article
        cleaned_article = {
            'article_number': article_num,
            'title': title,
            'source': source,
            'full_text': full_text,
            'clauses': clauses,
            'keywords': list(all_keywords),
            'num_clauses': len(clauses)
        }
        
        cleaned_articles.append(cleaned_article)
        
        # Generate QA pairs
        # Basic article QA
        qa_pairs.append({
            'question': f"What is Article {article_num} about?",
            'answer': f"Article {article_num} is titled '{title}' and deals with {title.lower()}.",
            'context': full_text,
            'article_number': article_num,
            'type': 'general'
        })
        
        # Clause-specific QAs
        for clause in clauses:
            if clause['content'] and len(clause['content']) > 20:
                qa_pairs.append({
                    'question': f"What does clause {clause['clause']} of Article {article_num} state?",
                    'answer': clause['content'],
                    'context': full_text,
                    'article_number': article_num,
                    'clause': clause['clause'],
                    'type': 'clause_specific'
                })
        
        # Keyword-based QAs
        for keyword in list(all_keywords)[:3]:  # Limit to top 3 keywords
            if keyword:
                qa_pairs.append({
                    'question': f"Which article deals with {keyword}?",
                    'answer': f"Article {article_num} ({title}) deals with {keyword}.",
                    'context': full_text,
                    'article_number': article_num,
                    'keyword': keyword,
                    'type': 'keyword_based'
                })
    
    # Export data in different formats
    print(f"\\nProcessed {len(cleaned_articles)} articles and generated {len(qa_pairs)} QA pairs")
    
    # 1. Structured JSON
    with open('constitution_structured.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_articles, f, indent=2, ensure_ascii=False)
    
    # 2. QA pairs JSON
    with open('constitution_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    # 3. Training JSONL format
    with open('constitution_training.jsonl', 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\\n')
    
    # 4. Plain text for language models
    with open('constitution_plain_text.txt', 'w', encoding='utf-8') as f:
        for article in cleaned_articles:
            f.write(f"\\n{'='*60}\\n")
            f.write(article['full_text'])
            f.write(f"\\n{'='*60}\\n")
    
    # 5. CSV formats
    try:
        # Articles CSV
        df_articles = pd.DataFrame([
            {
                'article_number': art['article_number'],
                'title': art['title'],
                'source': art['source'],
                'keywords': ', '.join(art['keywords']),
                'num_clauses': art['num_clauses'],
                'text_length': len(art['full_text'])
            }
            for art in cleaned_articles
        ])
        df_articles.to_csv('constitution_articles.csv', index=False, encoding='utf-8')
        
        # QA pairs CSV
        df_qa = pd.DataFrame(qa_pairs)
        df_qa.to_csv('constitution_qa_pairs.csv', index=False, encoding='utf-8')
        
    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
    
    # 6. Embeddings dataset
    embeddings_data = []
    for article in cleaned_articles:
        # Full article
        embeddings_data.append({
            'id': f"article_{article['article_number']}",
            'text': article['full_text'],
            'metadata': {
                'type': 'full_article',
                'article_number': article['article_number'],
                'title': article['title'],
                'keywords': article['keywords']
            }
        })
        
        # Individual clauses
        for i, clause in enumerate(article['clauses']):
            if clause['content'] and len(clause['content']) > 10:
                embeddings_data.append({
                    'id': f"article_{article['article_number']}_clause_{i}",
                    'text': clause['content'],
                    'metadata': {
                        'type': 'clause',
                        'article_number': article['article_number'],
                        'clause': clause['clause'],
                        'parent_title': article['title']
                    }
                })
    
    with open('constitution_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_clauses = sum(len(art['clauses']) for art in cleaned_articles)
    total_keywords = sum(len(art['keywords']) for art in cleaned_articles)
    
    print("\\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Total articles processed: {len(cleaned_articles)}")
    print(f"Total clauses: {total_clauses}")
    print(f"Total QA pairs generated: {len(qa_pairs)}")
    print(f"Total keywords: {total_keywords}")
    print(f"Embeddings entries: {len(embeddings_data)}")
    
    print("\\nFiles created:")
    print("- constitution_structured.json (structured article data)")
    print("- constitution_qa_pairs.json (Q&A pairs)")
    print("- constitution_training.jsonl (training format)")
    print("- constitution_plain_text.txt (plain text)")
    print("- constitution_articles.csv (articles summary)")
    print("- constitution_qa_pairs.csv (Q&A in CSV)")
    print("- constitution_embeddings.json (for vector databases)")
    
    print("\\nData is now ready for:")
    print("✓ Local LLM fine-tuning (use .jsonl file)")
    print("✓ RAG systems (use embeddings.json)")
    print("✓ Analysis & research (use .csv files)")
    print("✓ General NLP tasks (use structured.json)")

if __name__ == "__main__":
    print("Constitution Data Cleaner")
    print("=" * 25)
    process_constitution_data()