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
    
    # Remove multiple periods and asterisks
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\*+', '', text)
    
    # Remove footnote markers
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\d+\[\*+\]', '', text)
    
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
    
    # Extract important civil procedure concepts
    civil_patterns = [
        r'\b(court|judge|decree|judgment|order|suit|plaint|pleader|jurisdiction|appeal|execution|procedure)\b',
        r'\b(plaintiff|defendant|parties|application|petition|summons|service|property|damages|injunction)\b',
        r'\b(trial|hearing|evidence|witness|examination|cross-examination|argument|submission|ruling)\b'
    ]
    
    for pattern in civil_patterns:
        matches = re.findall(pattern, text.lower())
        important_words.update(matches)
    
    # Common stop words to exclude
    stop_words = {'this', 'that', 'with', 'from', 'they', 'been', 'have', 'their', 
                  'said', 'each', 'which', 'them', 'than', 'many', 'some', 'time',
                  'very', 'when', 'much', 'where', 'your', 'way', 'too', 'any',
                  'may', 'say', 'she', 'use', 'her', 'all', 'there', 'how', 'such',
                  'under', 'shall', 'being', 'other', 'code', 'section', 'chapter',
                  'means', 'includes', 'unless', 'provisions', 'force', 'government'}
    
    # Filter and clean keywords
    keywords = [word for word in important_words 
                if len(word) > 3 and word not in stop_words]
    
    return keywords[:10]  # Return top 10 keywords

def get_chapter_info(section_num: int) -> tuple:
    """Get appropriate chapter info based on CPC structure."""
    if section_num <= 5:
        return ("1", "Preliminary")
    elif section_num <= 25:
        return ("2", "Suits in General")
    elif section_num <= 35:
        return ("3", "Suits in Particular Cases")
    elif section_num <= 50:
        return ("4", "Institution of Suits")
    elif section_num <= 75:
        return ("5", "Issue and Service of Summons")
    elif section_num <= 82:
        return ("6", "Pleadings Generally")
    elif section_num <= 95:
        return ("7", "Parties to Suits")
    elif section_num <= 104:
        return ("8", "Suits by or against Government and Public Officers")
    elif section_num <= 115:
        return ("9", "Interpleader")
    elif section_num <= 125:
        return ("10", "Return of Summons")
    elif section_num <= 135:
        return ("11", "Appearance of Parties and Consequences of Non-appearance")
    elif section_num <= 148:
        return ("12", "Admission and Denial")
    elif section_num <= 153:
        return ("13", "Production, Impounding and Return of Documents")
    elif section_num <= 165:
        return ("14", "Settlement of Issues and Determination of Suit on Issues of Law or on Issues Agreed Upon")
    elif section_num <= 179:
        return ("15", "Disposal of Suit Otherwise than by Trial")
    elif section_num <= 186:
        return ("16", "Commencement and Conduct of Trial")
    elif section_num <= 195:
        return ("17", "Adjournment")
    elif section_num <= 214:
        return ("18", "Judgment and Decree")
    elif section_num <= 229:
        return ("19", "Costs")
    elif section_num <= 244:
        return ("20", "Appeals from Original Decrees")
    elif section_num <= 254:
        return ("21", "Appeals from Appellate Decrees")
    elif section_num <= 266:
        return ("22", "Special Cases")
    elif section_num <= 315:
        return ("23", "Execution of Decrees and Orders")
    else:
        return ("24", "Miscellaneous")

def process_cpc_data():
    """Process the CPC.json file and create cleaned datasets."""
    
    # Check if CPC.json exists
    if not os.path.exists('cpc.json'):
        print("Error: CPC.json file not found in current directory!")
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        return
    
    # Load the data
    print("Loading CPC.json...")
    with open('CPC.json', 'r', encoding='utf-8') as f:
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
        
        # Get chapter info
        try:
            section_int = int(section_num.split('A')[0].split('B')[0])  # Handle sections like 115A
        except:
            section_int = 1
            
        chapter, chapter_title = get_chapter_info(section_int)
        chapters[chapter] = chapter_title
        
        # Extract keywords
        keywords = extract_keywords(description, title)
        
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
            'question': f"What is Section {section_num} of CPC about?",
            'answer': f"Section {section_num} is titled '{title}' and deals with: {description[:200]}{'...' if len(description) > 200 else ''}",
            'context': full_text,
            'section_number': section_num,
            'chapter': chapter,
            'type': 'general'
        })
        
        # Title-based QA
        if title:
            qa_pairs.append({
                'question': f"Which section of CPC deals with {title.lower()}?",
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
                    'question': f"Which section of CPC deals with {keyword}?",
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
                'question': f"What sections are in Chapter {chapter} of CPC about {chapter_title}?",
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
                    'question': f"What does '{term}' mean in CPC?",
                    'answer': f"According to Section {section_num}, '{term}' means {definition.strip()}",
                    'context': full_text,
                    'section_number': section_num,
                    'term': term,
                    'type': 'definition_based'
                })
        
        # Procedure-based QAs (specific to CPC)
        if any(word in title.lower() for word in ['procedure', 'suit', 'appeal', 'execution', 'trial', 'service', 'summons']):
            qa_pairs.append({
                'question': f"What is the procedure for {title.lower()}?",
                'answer': f"Section {section_num} outlines the procedure for {title.lower()}: {description[:200]}{'...' if len(description) > 200 else ''}",
                'context': full_text,
                'section_number': section_num,
                'chapter': chapter,
                'type': 'procedure_based'
            })
        
        # Civil law specific QAs
        if any(word in description.lower() for word in ['plaintiff', 'defendant', 'decree', 'judgment', 'damages']):
            qa_pairs.append({
                'question': f"How does Section {section_num} apply to civil litigation?",
                'answer': f"In civil litigation, Section {section_num} ({title}) provides that: {description[:180]}{'...' if len(description) > 180 else ''}",
                'context': full_text,
                'section_number': section_num,
                'chapter': chapter,
                'type': 'civil_litigation'
            })
    
    # Export data in different formats
    print(f"\nProcessed {len(cleaned_sections)} sections and generated {len(qa_pairs)} QA pairs")
    
    # 1. Structured JSON
    with open('cpc_structured.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_sections, f, indent=2, ensure_ascii=False)
    
    # 2. QA pairs JSON
    with open('cpc_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    # 3. Training JSONL format
    with open('cpc_training.jsonl', 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # 4. Plain text for language models
    with open('cpc_plain_text.txt', 'w', encoding='utf-8') as f:
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
        df_sections.to_csv('cpc_sections.csv', index=False, encoding='utf-8')
        
        # QA pairs CSV
        df_qa = pd.DataFrame(qa_pairs)
        df_qa.to_csv('cpc_qa_pairs.csv', index=False, encoding='utf-8')
        
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
        df_chapters.to_csv('cpc_chapters.csv', index=False, encoding='utf-8')
        
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
    
    with open('cpc_embeddings.json', 'w', encoding='utf-8') as f:
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
    
    with open('cpc_by_chapters.json', 'w', encoding='utf-8') as f:
        json.dump(chapters_organized, f, indent=2, ensure_ascii=False)
    
    # 8. Definitions extraction (special for CPC)
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
        with open('cpc_definitions.json', 'w', encoding='utf-8') as f:
            json.dump(definitions, f, indent=2, ensure_ascii=False)
        
        # Also create a definitions CSV
        try:
            df_definitions = pd.DataFrame(definitions)
            df_definitions.to_csv('cpc_definitions.csv', index=False, encoding='utf-8')
        except:
            pass
    
    # 9. Civil procedure types classification
    procedure_types = {
        'suits': [],
        'appeals': [],
        'execution': [],
        'summons': [],
        'pleadings': [],
        'judgment': [],
        'decree': [],
        'costs': [],
        'evidence': [],
        'jurisdiction': [],
        'parties': [],
        'service': []
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
    
    with open('cpc_procedure_types.json', 'w', encoding='utf-8') as f:
        json.dump(procedure_types, f, indent=2, ensure_ascii=False)
    
    # 10. Court hierarchy and jurisdiction analysis
    court_related_sections = []
    for section in cleaned_sections:
        if any(term in section['description'].lower() for term in ['court', 'judge', 'jurisdiction', 'district court', 'high court']):
            court_related_sections.append({
                'section_number': section['section_number'],
                'title': section['title'],
                'chapter': section['chapter'],
                'relevance': 'court_jurisdiction'
            })
    
    with open('cpc_court_hierarchy.json', 'w', encoding='utf-8') as f:
        json.dump(court_related_sections, f, indent=2, ensure_ascii=False)
    
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
    print(f"Court-related sections: {len(court_related_sections)}")
    
    print("\nFiles created:")
    print("- cpc_structured.json (structured section data)")
    print("- cpc_qa_pairs.json (Q&A pairs)")
    print("- cpc_training.jsonl (training format)")
    print("- cpc_plain_text.txt (plain text with chapters)")
    print("- cpc_sections.csv (sections summary)")
    print("- cpc_chapters.csv (chapters summary)")
    print("- cpc_qa_pairs.csv (Q&A in CSV)")
    print("- cpc_embeddings.json (for vector databases)")
    print("- cpc_by_chapters.json (organized by chapters)")
    print("- cpc_procedure_types.json (categorized by procedure types)")
    print("- cpc_court_hierarchy.json (court jurisdiction sections)")
    if definitions:
        print("- cpc_definitions.json (extracted definitions)")
        print("- cpc_definitions.csv (definitions in CSV)")
    
    print("\nData is now ready for:")
    print("✓ Legal AI systems (use embeddings.json)")
    print("✓ Civil procedure research (use procedure_types.json)")
    print("✓ Court jurisdiction analysis (use court_hierarchy.json)")
    print("✓ Local LLM fine-tuning (use .jsonl file)")
    print("✓ RAG systems for CPC queries (use embeddings.json)")
    print("✓ Legal research & analysis (use .csv files)")
    print("✓ Chapter-wise study (use by_chapters.json)")
    
    # Print sample chapter info
    print(f"\nFirst 12 chapters:")
    for chapter_num in sorted([int(k) for k in list(chapters.keys())])[:12]:
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
    print("Code of Civil Procedure (CPC) Data Processor")
    print("=" * 48)
    process_cpc_data()