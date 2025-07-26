import json
import re
import os
from typing import Dict, List, Any
import pandas as pd

def natural_sort_key(section_num: str) -> tuple:
    """Create a sort key for alphanumeric section numbers like 2A, 2B, etc."""
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

def extract_keywords(text: str, title: str) -> List[str]:
    """Extract keywords from text and title for better search capability."""
    keywords = []
    
    # Common legal terms and important words
    important_words = set()
    
    # Extract from title (more important)
    title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
    important_words.update(title_words)
    
    # Extract key terms from description
    # Look for definitions and important legal terms
    definition_matches = re.findall(r'"([^"]+)"', text)
    for match in definition_matches:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', match.lower())
        important_words.update(words[:3])  # Limit to avoid too many keywords
    
    # Extract capitalized terms (likely important legal concepts)
    cap_terms = re.findall(r'\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*\b', text)
    for term in cap_terms[:5]:  # Limit to top 5
        important_words.add(term.lower())
    
    # Common stop words to exclude
    stop_words = {'this', 'that', 'with', 'from', 'they', 'been', 'have', 'their', 
                  'said', 'each', 'which', 'them', 'than', 'many', 'some', 'time',
                  'very', 'when', 'much', 'where', 'your', 'way', 'too', 'any',
                  'may', 'say', 'she', 'use', 'her', 'all', 'there', 'how'}
    
    # Filter and clean keywords
    keywords = [word for word in important_words 
                if len(word) > 3 and word not in stop_words]
    
    return keywords[:10]  # Return top 10 keywords

def process_mva_data():
    """Process the MVA.json file and create cleaned datasets."""
    
    # Check if MVA.json exists
    if not os.path.exists('MVA.json'):
        print("Error: MVA.json file not found in current directory!")
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        return
    
    # Load the data
    print("Loading MVA.json...")
    with open('MVA.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} sections")
    
    # Process sections
    cleaned_sections = []
    qa_pairs = []
    chapters = {}
    
    for item in raw_data:
        # Extract section info
        section_num = str(item.get('section', '')).strip()
        title = clean_text(item.get('title', ''))
        description = clean_text(item.get('description', ''))
        
        # Extract keywords
        keywords = extract_keywords(description, title)
        
        # Try to determine chapter from context (MVA doesn't have explicit chapters)
        # We'll group by section number ranges for organization
        section_int = 0
        try:
            section_match = re.match(r'(\d+)', section_num)
            if section_match:
                section_int = int(section_match.group(1))
        except:
            pass
        
        # Group into logical chapters based on section ranges
        if section_int <= 10:
            chapter = "1"
            chapter_title = "Preliminary"
        elif section_int <= 30:
            chapter = "2"
            chapter_title = "Licensing of Drivers"
        elif section_int <= 60:
            chapter = "3"
            chapter_title = "Licensing of Conductors"
        elif section_int <= 90:
            chapter = "4"
            chapter_title = "Registration of Motor Vehicles"
        elif section_int <= 120:
            chapter = "5"
            chapter_title = "Control of Transport Vehicles"
        elif section_int <= 150:
            chapter = "6"
            chapter_title = "Special Provisions"
        elif section_int <= 180:
            chapter = "7"
            chapter_title = "Construction and Maintenance"
        elif section_int <= 210:
            chapter = "8"
            chapter_title = "Motor Vehicle Testing"
        else:
            chapter = "9"
            chapter_title = "Offences and Penalties"
        
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
            'question': f"What is Section {section_num} of Motor Vehicles Act about?",
            'answer': f"Section {section_num} is titled '{title}' and states: {description[:200]}{'...' if len(description) > 200 else ''}",
            'context': full_text,
            'section_number': section_num,
            'chapter': chapter,
            'type': 'general'
        })
        
        # Title-based QA
        if title:
            qa_pairs.append({
                'question': f"Which section of MVA deals with {title.lower()}?",
                'answer': f"Section {section_num} deals with {title.lower()}. {description[:150]}{'...' if len(description) > 150 else ''}",
                'context': full_text,
                'section_number': section_num,
                'chapter': chapter,
                'type': 'title_based'
            })
        
        # Keyword-based QAs
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            if keyword and len(keyword) > 3:
                qa_pairs.append({
                    'question': f"Which section of Motor Vehicles Act deals with {keyword}?",
                    'answer': f"Section {section_num} ({title}) deals with {keyword}. {description[:150]}{'...' if len(description) > 150 else ''}",
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
        
        # Definition-based QAs (for sections with definitions)
        if 'definition' in title.lower() or '"' in description:
            definition_matches = re.findall(r'"([^"]+)"[^"]*means\s+([^;]+)', description)
            for term, definition in definition_matches[:3]:  # Limit to avoid too many
                qa_pairs.append({
                    'question': f"What does '{term}' mean in Motor Vehicles Act?",
                    'answer': f"According to Section {section_num}, '{term}' means {definition.strip()}",
                    'context': full_text,
                    'section_number': section_num,
                    'term': term,
                    'type': 'definition_based'
                })
    
    # Export data in different formats
    print(f"\nProcessed {len(cleaned_sections)} sections and generated {len(qa_pairs)} QA pairs")
    
    # 1. Structured JSON
    with open('mva_structured.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_sections, f, indent=2, ensure_ascii=False)
    
    # 2. QA pairs JSON
    with open('mva_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    # 3. Training JSONL format
    with open('mva_training.jsonl', 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # 4. Plain text for language models
    with open('mva_plain_text.txt', 'w', encoding='utf-8') as f:
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
                'description': sec['description'][:500] + ('...' if len(sec['description']) > 500 else ''),  # Truncate for CSV
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
        df_sections.to_csv('mva_sections.csv', index=False, encoding='utf-8')
        
        # QA pairs CSV
        df_qa = pd.DataFrame(qa_pairs)
        df_qa.to_csv('mva_qa_pairs.csv', index=False, encoding='utf-8')
        
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
        df_chapters.to_csv('mva_chapters.csv', index=False, encoding='utf-8')
        
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
        if len(section['description']) > 100:  # Only for substantial descriptions
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
    
    with open('mva_embeddings.json', 'w', encoding='utf-8') as f:
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
    
    with open('mva_by_chapters.json', 'w', encoding='utf-8') as f:
        json.dump(chapters_organized, f, indent=2, ensure_ascii=False)
    
    # 8. Definitions extraction (special for MVA)
    definitions = []
    for section in cleaned_sections:
        if 'definition' in section['title'].lower() or '"' in section['description']:
            # Extract definitions using regex
            definition_matches = re.findall(r'"([^"]+)"[^"]*means\s+([^;]+)', section['description'])
            for term, definition in definition_matches:
                definitions.append({
                    'term': term.strip(),
                    'definition': definition.strip(),
                    'section_number': section['section_number'],
                    'section_title': section['title']
                })
    
    if definitions:
        with open('mva_definitions.json', 'w', encoding='utf-8') as f:
            json.dump(definitions, f, indent=2, ensure_ascii=False)
        
        # Also create a definitions CSV
        try:
            df_definitions = pd.DataFrame(definitions)
            df_definitions.to_csv('mva_definitions.csv', index=False, encoding='utf-8')
        except:
            pass
    
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
    print(f"Definitions extracted: {len(definitions)}")
    
    print("\nFiles created:")
    print("- mva_structured.json (structured section data)")
    print("- mva_qa_pairs.json (Q&A pairs)")
    print("- mva_training.jsonl (training format)")
    print("- mva_plain_text.txt (plain text with chapters)")
    print("- mva_sections.csv (sections summary)")
    print("- mva_chapters.csv (chapters summary)")
    print("- mva_qa_pairs.csv (Q&A in CSV)")
    print("- mva_embeddings.json (for vector databases)")
    print("- mva_by_chapters.json (organized by chapters)")
    if definitions:
        print("- mva_definitions.json (extracted definitions)")
        print("- mva_definitions.csv (definitions in CSV)")
    
    print("\nData is now ready for:")
    print("✓ Legal AI systems (use embeddings.json)")
    print("✓ Motor vehicle law research (use definitions.json)")
    print("✓ Local LLM fine-tuning (use .jsonl file)")
    print("✓ RAG systems for MVA queries (use embeddings.json)")
    print("✓ Legal research & analysis (use .csv files)")
    print("✓ Chapter-wise study (use by_chapters.json)")
    
    # Print sample chapter info
    print(f"\nChapter organization:")
    for chapter_num in sorted(list(chapters.keys())):
        chapter_sections = [s for s in cleaned_sections if s['chapter'] == chapter_num]
        print(f"- Chapter {chapter_num}: {chapters[chapter_num]} ({len(chapter_sections)} sections)")
    
    # Print some extracted definitions
    if definitions:
        print(f"\nSample definitions found:")
        for defn in definitions[:5]:
            print(f"- {defn['term']}: {defn['definition'][:80]}{'...' if len(defn['definition']) > 80 else ''}")

if __name__ == "__main__":
    print("Motor Vehicles Act (MVA) Data Processor")
    print("=" * 42)
    process_mva_data()