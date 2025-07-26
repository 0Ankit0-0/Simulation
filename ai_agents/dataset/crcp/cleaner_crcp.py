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
    
    # Remove extra whitespaces and tabs
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\t+', ' ', text)
    
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
    
    # Extract important legal concepts
    legal_patterns = [
        r'\b(magistrate|court|judge|sessions?|jurisdiction|offence|criminal|procedure|trial|arrest|warrant|bail|appeal|investigation|inquiry)\b',
        r'\b(punishment|sentence|fine|imprisonment|custody|detention|prosecution|defence|evidence|witness)\b'
    ]
    
    for pattern in legal_patterns:
        matches = re.findall(pattern, text.lower())
        important_words.update(matches)
    
    # Common stop words to exclude
    stop_words = {'this', 'that', 'with', 'from', 'they', 'been', 'have', 'their', 
                  'said', 'each', 'which', 'them', 'than', 'many', 'some', 'time',
                  'very', 'when', 'much', 'where', 'your', 'way', 'too', 'any',
                  'may', 'say', 'she', 'use', 'her', 'all', 'there', 'how', 'such',
                  'under', 'shall', 'being', 'other', 'code', 'section', 'chapter'}
    
    # Filter and clean keywords
    keywords = [word for word in important_words 
                if len(word) > 3 and word not in stop_words]
    
    return keywords[:10]  # Return top 10 keywords

def get_chapter_title(chapter_num: int) -> str:
    """Get appropriate chapter title based on CrPC structure."""
    chapter_titles = {
        1: "Preliminary",
        2: "Constitution of Criminal Courts and Offices",
        3: "Powers of Courts",
        4: "General Provisions as to Criminal Courts",
        5: "Jurisdiction of the Criminal Courts in Inquiries and Trials",
        6: "Of the Exercise of Powers of Executive Magistrates",
        7: "Processes to Compel Appearance",
        8: "Security for Keeping the Peace and for Good Behaviour",
        9: "Order for Maintenance of Wives, Children and Parents",
        10: "Public Nuisances",
        11: "Prevention of Offences",
        12: "Information to the Police and their Powers to Investigate",
        13: "Jurisdiction of the Criminal Courts in Inquiries and Trials",
        14: "General Provisions as to Inquiries and Trials",
        15: "Of Bail and Bonds",
        16: "Of Summons, Warrants and Proclamations",
        17: "Of the Processes to Compel the Production of Things",
        18: "Of Search-warrants",
        19: "Of Arrest of Persons",
        20: "Of Public Prosecutors",
        21: "The Inquiry",
        22: "The Trial",
        23: "Of the Judgment",
        24: "General Provisions as to Trials before Courts of Session",
        25: "Trial by Jury",
        26: "General Provisions as to Trials before Magistrates",
        27: "Summary Trials",
        28: "The Sentences",
        29: "Suspension, Remission and Commutation of Sentences",
        30: "Probation of Offenders",
        31: "Previous Conviction",
        32: "Disposal of Property",
        33: "Appeals",
        34: "Execution, Suspension, Remission and Commutation of Sentences",
        35: "Irregular Proceedings",
        36: "Limitation for Taking Cognizance of Offences",
        37: "Miscellaneous"
    }
    
    return chapter_titles.get(chapter_num, f"Chapter {chapter_num}")

def process_crpc_data():
    """Process the crcp.json file and create cleaned datasets."""
    
    # Check if crcp.json exists
    if not os.path.exists('crcp.json'):
        print("Error: crcp.json file not found in current directory!")
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        return
    
    # Load the data
    print("Loading crcp.json...")
    with open('crcp.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} sections")
    
    # Process sections
    cleaned_sections = []
    qa_pairs = []
    chapters = {}
    
    for item in raw_data:
        # Extract section info
        chapter_num = item.get('chapter', 0)
        section_num = str(item.get('section', '')).strip()
        title = clean_text(item.get('section_title', ''))
        description = clean_text(item.get('section_desc', ''))
        
        # Get chapter title
        chapter_title = get_chapter_title(chapter_num)
        chapters[str(chapter_num)] = chapter_title
        
        # Extract keywords
        keywords = extract_keywords(description, title)
        
        # Create full text
        full_text = f"Section {section_num}: {title}\n{description}"
        
        # Create cleaned section
        cleaned_section = {
            'section_number': section_num,
            'title': title,
            'description': description,
            'chapter': str(chapter_num),
            'chapter_title': chapter_title,
            'keywords': keywords,
            'full_text': full_text,
            'text_length': len(description)
        }
        
        cleaned_sections.append(cleaned_section)
        
        # Generate QA pairs
        # Basic section QA
        qa_pairs.append({
            'question': f"What is Section {section_num} of CrPC about?",
            'answer': f"Section {section_num} is titled '{title}' and deals with: {description[:200]}{'...' if len(description) > 200 else ''}",
            'context': full_text,
            'section_number': section_num,
            'chapter': str(chapter_num),
            'type': 'general'
        })
        
        # Title-based QA
        if title:
            qa_pairs.append({
                'question': f"Which section of CrPC deals with {title.lower()}?",
                'answer': f"Section {section_num} deals with {title.lower()}. {description[:150]}{'...' if len(description) > 150 else ''}",
                'context': full_text,
                'section_number': section_num,
                'chapter': str(chapter_num),
                'type': 'title_based'
            })
        
        # Keyword-based QAs
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            if keyword and len(keyword) > 3:
                qa_pairs.append({
                    'question': f"Which section of CrPC deals with {keyword}?",
                    'answer': f"Section {section_num} ({title}) deals with {keyword}. {description[:150]}{'...' if len(description) > 150 else ''}",
                    'context': full_text,
                    'section_number': section_num,
                    'keyword': keyword,
                    'chapter': str(chapter_num),
                    'type': 'keyword_based'
                })
        
        # Chapter-based QA
        if chapter_title:
            qa_pairs.append({
                'question': f"What sections are in Chapter {chapter_num} of CrPC about {chapter_title}?",
                'answer': f"Section {section_num} is part of Chapter {chapter_num} on {chapter_title}. This section covers: {title}",
                'context': full_text,
                'section_number': section_num,
                'chapter': str(chapter_num),
                'chapter_title': chapter_title,
                'type': 'chapter_based'
            })
        
        # Definition-based QAs (for sections with definitions)
        if 'definition' in title.lower() or '"' in description:
            definition_matches = re.findall(r'"([^"]+)"[^"]*means\s+([^;]+)', description)
            for term, definition in definition_matches[:3]:  # Limit to avoid too many
                qa_pairs.append({
                    'question': f"What does '{term}' mean in CrPC?",
                    'answer': f"According to Section {section_num}, '{term}' means {definition.strip()}",
                    'context': full_text,
                    'section_number': section_num,
                    'term': term,
                    'type': 'definition_based'
                })
        
        # Procedure-based QAs (specific to CrPC)
        if any(word in title.lower() for word in ['procedure', 'trial', 'inquiry', 'arrest', 'warrant', 'bail']):
            qa_pairs.append({
                'question': f"What is the procedure for {title.lower()}?",
                'answer': f"Section {section_num} outlines the procedure for {title.lower()}: {description[:200]}{'...' if len(description) > 200 else ''}",
                'context': full_text,
                'section_number': section_num,
                'chapter': str(chapter_num),
                'type': 'procedure_based'
            })
    
    # Export data in different formats
    print(f"\nProcessed {len(cleaned_sections)} sections and generated {len(qa_pairs)} QA pairs")
    
    # 1. Structured JSON
    with open('crpc_structured.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_sections, f, indent=2, ensure_ascii=False)
    
    # 2. QA pairs JSON
    with open('crpc_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    # 3. Training JSONL format
    with open('crpc_training.jsonl', 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # 4. Plain text for language models
    with open('crpc_plain_text.txt', 'w', encoding='utf-8') as f:
        current_chapter = None
        for section in sorted(cleaned_sections, key=lambda x: (int(x['chapter']), natural_sort_key(x['section_number']))):
            # Add chapter header if new chapter
            if section['chapter'] != current_chapter:
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
        # Sort by chapter and section
        df_sections['chapter_int'] = df_sections['chapter'].astype(int)
        df_sections['sort_key'] = df_sections['section_number'].apply(natural_sort_key)
        df_sections = df_sections.sort_values(['chapter_int', 'sort_key']).drop(['chapter_int', 'sort_key'], axis=1)
        df_sections.to_csv('crpc_sections.csv', index=False, encoding='utf-8')
        
        # QA pairs CSV
        df_qa = pd.DataFrame(qa_pairs)
        df_qa.to_csv('crpc_qa_pairs.csv', index=False, encoding='utf-8')
        
        # Chapters summary CSV
        chapters_summary = []
        for chapter_num, chapter_title in chapters.items():
            chapter_sections = [s for s in cleaned_sections if s['chapter'] == chapter_num]
            if chapter_sections:
                min_section = min(chapter_sections, key=lambda x: natural_sort_key(x['section_number']))['section_number']
                max_section = max(chapter_sections, key=lambda x: natural_sort_key(x['section_number']))['section_number']
                chapters_summary.append({
                    'chapter_number': int(chapter_num),
                    'chapter_title': chapter_title,
                    'total_sections': len(chapter_sections),
                    'section_range': f"{min_section}-{max_section}"
                })
        
        df_chapters = pd.DataFrame(chapters_summary)
        df_chapters = df_chapters.sort_values('chapter_number')
        df_chapters.to_csv('crpc_chapters.csv', index=False, encoding='utf-8')
        
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
    
    with open('crpc_embeddings.json', 'w', encoding='utf-8') as f:
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
    
    # Sort sections within each chapter
    for chapter_data in chapters_organized.values():
        chapter_data['sections'].sort(key=lambda x: natural_sort_key(x['section_number']))
    
    with open('crpc_by_chapters.json', 'w', encoding='utf-8') as f:
        json.dump(chapters_organized, f, indent=2, ensure_ascii=False)
    
    # 8. Definitions extraction (special for CrPC)
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
                    'section_title': section['title'],
                    'chapter': section['chapter'],
                    'chapter_title': section['chapter_title']
                })
    
    if definitions:
        with open('crpc_definitions.json', 'w', encoding='utf-8') as f:
            json.dump(definitions, f, indent=2, ensure_ascii=False)
        
        # Also create a definitions CSV
        try:
            df_definitions = pd.DataFrame(definitions)
            df_definitions.to_csv('crpc_definitions.csv', index=False, encoding='utf-8')
        except:
            pass
    
    # 9. Procedure types classification
    procedure_types = {
        'arrest': [],
        'bail': [],
        'trial': [],
        'inquiry': [],
        'investigation': [],
        'warrant': [],
        'appeal': [],
        'magistrate': [],
        'court': [],
        'sentence': []
    }
    
    for section in cleaned_sections:
        title_lower = section['title'].lower()
        desc_lower = section['description'].lower()
        
        for proc_type in procedure_types.keys():
            if proc_type in title_lower or proc_type in desc_lower:
                procedure_types[proc_type].append({
                    'section_number': section['section_number'],
                    'title': section['title'],
                    'chapter': section['chapter']
                })
    
    with open('crpc_procedure_types.json', 'w', encoding='utf-8') as f:
        json.dump(procedure_types, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_keywords = sum(len(sec['keywords']) for sec in cleaned_sections)
    avg_section_length = sum(sec['text_length'] for sec in cleaned_sections) / len(cleaned_sections)
    total_chapters = len(chapters)
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Total sections processed: {len(cleaned_sections)}")
    print(f"Total chapters: {total_chapters}")
    print(f"Total QA pairs generated: {len(qa_pairs)}")
    print(f"Total keywords: {total_keywords}")
    print(f"Embeddings entries: {len(embeddings_data)}")
    print(f"Average section length: {avg_section_length:.0f} characters")
    print(f"Definitions extracted: {len(definitions)}")
    
    print("\nFiles created:")
    print("- crpc_structured.json (structured section data)")
    print("- crpc_qa_pairs.json (Q&A pairs)")
    print("- crpc_training.jsonl (training format)")
    print("- crpc_plain_text.txt (plain text with chapters)")
    print("- crpc_sections.csv (sections summary)")
    print("- crpc_chapters.csv (chapters summary)")
    print("- crpc_qa_pairs.csv (Q&A in CSV)")
    print("- crpc_embeddings.json (for vector databases)")
    print("- crpc_by_chapters.json (organized by chapters)")
    print("- crpc_procedure_types.json (categorized by procedure types)")
    if definitions:
        print("- crpc_definitions.json (extracted definitions)")
        print("- crpc_definitions.csv (definitions in CSV)")
    
    print("\nData is now ready for:")
    print("✓ Legal AI systems (use embeddings.json)")
    print("✓ Criminal procedure research (use procedure_types.json)")
    print("✓ Local LLM fine-tuning (use .jsonl file)")
    print("✓ RAG systems for CrPC queries (use embeddings.json)")
    print("✓ Legal research & analysis (use .csv files)")
    print("✓ Chapter-wise study (use by_chapters.json)")
    
    # Print sample chapter info
    print(f"\nFirst 10 chapters:")
    for chapter_num in sorted([int(k) for k in list(chapters.keys())])[:10]:
        chapter_sections = [s for s in cleaned_sections if s['chapter'] == str(chapter_num)]
        print(f"- Chapter {chapter_num}: {chapters[str(chapter_num)]} ({len(chapter_sections)} sections)")
    
    # Print some extracted definitions
    if definitions:
        print(f"\nSample definitions found:")
        for defn in definitions[:5]:
            print(f"- {defn['term']}: {defn['definition'][:80]}{'...' if len(defn['definition']) > 80 else ''}")
    
    # Print procedure type statistics
    print(f"\nProcedure type analysis:")
    for proc_type, sections in procedure_types.items():
        if sections:
            print(f"- {proc_type}: {len(sections)} sections")

if __name__ == "__main__":
    print("Code of Criminal Procedure (CrPC) Data Processor")
    print("=" * 50)
    process_crpc_data()