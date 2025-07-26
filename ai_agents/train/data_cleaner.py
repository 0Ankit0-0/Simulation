# import json
# import re
# from collections import Counter
# from nltk.corpus import stopwords
# import nltk

# nltk.download('stopwords')

# def clean_ipc(input_path, output_path):
#     with open(input_path, "r", encoding="utf-8") as f:
#         raw_data = json.load(f)

#     cleaned = {}
#     for entry in raw_data:
#         section_num = str(entry["Section"])
#         title = entry.get("section_title", "")
#         desc = entry.get("section_desc", "")

#         combined_text = f"{title} {desc}"
#         keywords = extract_keywords(combined_text)

#         cleaned[section_num] = {
#             "title": title,
#             "description": desc,
#             "chapter": entry.get("chapter"),
#             "chapter_title": entry.get("chapter_title"),
#             "keywords": keywords
#         }
#         cleaned[section_num] = {
#             "title": title,
#             "description": desc,
#             "chapter": entry.get("chapter"),
#             "chapter_title": entry.get("chapter_title"),
#             "keywords": keywords
#         }

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(cleaned, f, indent=2)

# def extract_keywords(text):
#     words = re.findall(r'\b[a-z]{4,}\b', text.lower())
#     stop_words = set(stopwords.words('english'))
#     keywords = [w for w in words if w not in stop_words]
#     common = Counter(keywords).most_common(6)
#     return [word for word, _ in common]

# # Example call:
# # clean_ipc("ai_agents/data/raw_ipc.json", "ai_agents/data/ipc.json")

# clean_ipc("ai_agents/data/raw_ipc.json", "ai_agents/dataset/ipc.json")

import json
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def clean_kaggle_ipc(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned = {}
    for entry in raw_data:
        section_id = entry.get("Section", "").replace("IPC_", "")
        offense = entry.get("Offense", "")
        description = entry.get("Description", "")
        punishment = entry.get("Punishment", "")

        if not section_id:
            continue

        combined_text = f"{offense} {description}"
        keywords = extract_keywords(combined_text)

        cleaned[section_id] = {
            "title": offense,
            "description": description,
            "punishment": punishment,
            "keywords": keywords
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)

def extract_keywords(text):
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    stop_words = set(stopwords.words('english'))
    keywords = [w for w in words if w not in stop_words]
    common = Counter(keywords).most_common(6)
    return [word for word, _ in common]

# Example usage
clean_kaggle_ipc("ai_agents/data/raw_ipc2.json", "ai_agents/dataset/k_ipc.json")
