import json
import re
import os
from typing import Dict, List, Any
import pandas as pd

def natural_sort_key(section_num: str) -> tuple:
    """Create a sort key for alphanumeric section numbers like 29A, 124A, etc."""
    import re
    # Split into numeric and alphabetic parts
    match = re.match(r'(\d+)([A-Za-z]*)', str(section_num))
    if match:
        num_part = int(match.group(1))
        alpha_part = match.group(2) if match.group(2) else ''
        return (num_part, alpha_part)
    else:
        # Fallback for unexpected formats
        return (0, str(section_num))

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

def process_ipc_data():
    """Process the IPC.json file and create cleaned datasets."""
    
    # Check if IPC.json exists
    if not os.path.exists('ipc.json'):
        print("Error: ipc.json file not found in current directory!")
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        return
    
    # Load the data
    print("Loading ipc.json...")
    with open('ipc.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} sections")
    
    # Process sections
    cleaned_sections = []
    qa_pairs = []
    chapters = {}
    
    for section_num, section_data in raw_data.items():
        # Extract section info
        title = clean_text(section_data.get('title', ''))
        description = clean_text(section_data.get('description', ''))
        chapter = section_data.get('chapter', '')
        chapter_title = clean_text(section_data.get('chapter_title', ''))
        keywords = section_data.get('keywords', [])
        
        # Track chapters
        if chapter and chapter_title:
            chapters[chapter] = chapter_title
        
        # Create full text
        full_text = f"Section {section_num}: {title}\n{description}"
        
        # Create cleaned section
        cleaned_section = {
            'section_number': section_num,
            'title': title,
            'description': description,
            'chapter': chapter,
            'chapter_title': chapter_title,
            'keywords': keywords,
            'full_text': full_text,
            'text_length': len(description)
        }
        
        cleaned_sections.append(cleaned_section)
        
        # Generate QA pairs
        # Basic section QA
        qa_pairs.append({
            'question': f"What is Section {section_num} of IPC about?",
            'answer': f"Section {section_num} is titled '{title}' and states: {description}",
            'context': full_text,
            'section_number': section_num,
            'chapter': chapter,
            'type': 'general'
        })
        
        # Title-based QA
        if title:
            qa_pairs.append({
                'question': f"Which section deals with {title.lower()}?",
                'answer': f"Section {section_num} deals with {title.lower()}. It states: {description}",
                'context': full_text,
                'section_number': section_num,
                'chapter': chapter,
                'type': 'title_based'
            })
        
        # Keyword-based QAs
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            if keyword and len(keyword) > 3:  # Skip very short keywords
                qa_pairs.append({
                    'question': f"Which section of IPC deals with {keyword}?",
                    'answer': f"Section {section_num} ({title}) deals with {keyword}. {description}",
                    'context': full_text,
                    'section_number': section_num,
                    'keyword': keyword,
                    'chapter': chapter,
                    'type': 'keyword_based'
                })
        
        # Chapter-based QA
        if chapter_title:
            qa_pairs.append({
                'question': f"What sections are in Chapter {chapter} about {chapter_title}?",
                'answer': f"Section {section_num} is part of Chapter {chapter} on {chapter_title}. This section covers: {title}",
                'context': full_text,
                'section_number': section_num,
                'chapter': chapter,
                'chapter_title': chapter_title,
                'type': 'chapter_based'
            })
    
    # Export data in different formats
    print(f"\nProcessed {len(cleaned_sections)} sections and generated {len(qa_pairs)} QA pairs")
    
    # 1. Structured JSON
    with open('ipc_structured.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_sections, f, indent=2, ensure_ascii=False)
    
    # 2. QA pairs JSON
    with open('ipc_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    # 3. Training JSONL format
    with open('ipc_training.jsonl', 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # 4. Plain text for language models
    with open('ipc_plain_text.txt', 'w', encoding='utf-8') as f:
        current_chapter = None
        for section in sorted(cleaned_sections, key=lambda x: natural_sort_key(x['section_number'])):
            # Add chapter header if new chapter
            if section['chapter'] != current_chapter and section['chapter_title']:
                current_chapter = section['chapter']
                f.write(f"\n{'='*80}\n")
                f.write(f"CHAPTER {current_chapter}: {section['chapter_title'].upper()}\n")
                f.write(f"{'='*80}\n\n")
            
            f.write(f"Section {section['section_number']}: {section['title']}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{section['description']}\n\n")
    
    # 5. CSV formats
    try:
        # Sections CSV
        df_sections = pd.DataFrame([
            {
                'section_number': sec['section_number'],
                'title': sec['title'],
                'description': sec['description'],
                'chapter': sec['chapter'],
                'chapter_title': sec['chapter_title'],
                'keywords': ', '.join(sec['keywords']),
                'text_length': sec['text_length']
            }
            for sec in cleaned_sections
        ])
        # Sort using natural sort key
        df_sections['sort_key'] = df_sections['section_number'].apply(natural_sort_key)
        df_sections = df_sections.sort_values('sort_key').drop('sort_key', axis=1)
        df_sections.to_csv('ipc_sections.csv', index=False, encoding='utf-8')
        
        # QA pairs CSV
        df_qa = pd.DataFrame(qa_pairs)
        df_qa.to_csv('ipc_qa_pairs.csv', index=False, encoding='utf-8')
        
        # Chapters summary CSV
        chapters_summary = []
        for chapter_num, chapter_title in chapters.items():
            chapter_sections = [s for s in cleaned_sections if s['chapter'] == chapter_num]
            chapters_summary.append({
                'chapter_number': chapter_num,
                'chapter_title': chapter_title,
                'total_sections': len(chapter_sections),
                'section_range': f"{min(chapter_sections, key=lambda x: natural_sort_key(x['section_number']))['section_number']}-{max(chapter_sections, key=lambda x: natural_sort_key(x['section_number']))['section_number']}" if chapter_sections else ""
            })
        
        df_chapters = pd.DataFrame(chapters_summary)
        df_chapters = df_chapters.sort_values('chapter_number')
        df_chapters.to_csv('ipc_chapters.csv', index=False, encoding='utf-8')
        
    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
    
    # 6. Embeddings dataset
    embeddings_data = []
    for section in cleaned_sections:
        # Full section
        embeddings_data.append({
            'id': f"section_{section['section_number']}",
            'text': section['full_text'],
            'metadata': {
                'type': 'full_section',
                'section_number': section['section_number'],
                'title': section['title'],
                'chapter': section['chapter'],
                'chapter_title': section['chapter_title'],
                'keywords': section['keywords']
            }
        })
        
        # Title and description separately for better granularity
        if len(section['description']) > 50:  # Only for substantial descriptions
            embeddings_data.append({
                'id': f"section_{section['section_number']}_desc",
                'text': section['description'],
                'metadata': {
                    'type': 'section_description',
                    'section_number': section['section_number'],
                    'title': section['title'],
                    'chapter': section['chapter'],
                    'chapter_title': section['chapter_title']
                }
            })
    
    with open('ipc_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    
    # 7. Chapter-wise organization
    chapters_organized = {}
    for section in cleaned_sections:
        chapter_key = f"chapter_{section['chapter']}"
        if chapter_key not in chapters_organized:
            chapters_organized[chapter_key] = {
                'chapter_number': section['chapter'],
                'chapter_title': section['chapter_title'],
                'sections': []
            }
        chapters_organized[chapter_key]['sections'].append(section)
    
    with open('ipc_by_chapters.json', 'w', encoding='utf-8') as f:
        json.dump(chapters_organized, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_keywords = sum(len(sec['keywords']) for sec in cleaned_sections)
    avg_section_length = sum(sec['text_length'] for sec in cleaned_sections) / len(cleaned_sections)
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Total sections processed: {len(cleaned_sections)}")
    print(f"Total chapters: {len(chapters)}")
    print(f"Total QA pairs generated: {len(qa_pairs)}")
    print(f"Total keywords: {total_keywords}")
    print(f"Embeddings entries: {len(embeddings_data)}")
    print(f"Average section length: {avg_section_length:.0f} characters")
    
    print("\nFiles created:")
    print("- ipc_structured.json (structured section data)")
    print("- ipc_qa_pairs.json (Q&A pairs)")
    print("- ipc_training.jsonl (training format)")
    print("- ipc_plain_text.txt (plain text with chapters)")
    print("- ipc_sections.csv (sections summary)")
    print("- ipc_chapters.csv (chapters summary)")
    print("- ipc_qa_pairs.csv (Q&A in CSV)")
    print("- ipc_embeddings.json (for vector databases)")
    print("- ipc_by_chapters.json (organized by chapters)")
    
    print("\nData is now ready for:")
    print("✓ Legal AI systems (use embeddings.json)")
    print("✓ Local LLM fine-tuning (use .jsonl file)")
    print("✓ RAG systems for legal queries (use embeddings.json)")
    print("✓ Legal research & analysis (use .csv files)")
    print("✓ Chapter-wise study (use by_chapters.json)")
    
    # Print sample chapter info
    print(f"\nSample chapters found:")
    for chapter_num in sorted(list(chapters.keys())[:5]):
        chapter_sections = [s for s in cleaned_sections if s['chapter'] == chapter_num]
        print(f"- Chapter {chapter_num}: {chapters[chapter_num]} ({len(chapter_sections)} sections)")

if __name__ == "__main__":
    print("Indian Penal Code (IPC) Data Processor")
    print("=" * 40)
    process_ipc_data()