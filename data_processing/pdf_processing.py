import fitz
import re
import csv
import os

def replace_equations(text):
    # Find equations that start with ":" and end with "(number)"
    equation_pattern = re.compile(r":[\s\S]{0,150}?\(\d{1,2}\)", re.DOTALL)

    # Replace equations with "(equation)"
    cleaned_text = re.sub(equation_pattern, r" (equation)", text)

    return cleaned_text

def replace_non_ascii(text):
    # Replace non-extended ASCII characters with (s)
    modified_text = "".join("(s)" if ord(char) > 255 else char for char in text)
    return modified_text


def extract_sections(blocks):
    sections = []
    current_section = None
    buffer = []
    patterns = [
        # Looks for the sections that are are not usually accompanied by a number
        r'(?i)(ABSTRACT|REFERENCES|BIBLIOGRAPHY|ETHICS STATEMENT|REPRODUCIBILITY STATEMENT|AUTHOR CONTRIBUTIONS)',
       
        # Looks for number-word pairs: 1 Introduction, 2. Methods etc.
        r'(?:^|\n)(\d+(?:\.\d+)*\.?)\s*\n?([A-Za-z&\-][A-Za-z&\s\-]*)(?=\n|$)',
    ]

    # Iterate through all pages
    for block in blocks:
        text = block[4].strip()
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                # Save previous section's content
                if current_section:
                    sections.append((current_section, buffer))
                # Start a new section

                if pattern == patterns[0]:
                    current_section = match.group(1).strip().upper()
                else:
                    current_section = match.group(2).strip().upper()
               
                buffer = []

                break 
       
        # Add text to the current section only if it's not a section heading
        if current_section:
            buffer.append(text)
       
    if current_section:
        sections.append((current_section, buffer))

    # Merge small paragraphs within sections
    merged_sections = []
    paragraph_count = 0
    for title, paragraphs in sections:
        merged_paragraphs = []

        for paragraph in paragraphs:

            if merged_paragraphs and len(paragraph) < 6:
                # Merge with previous paragraph
                merged_paragraphs[-1] += " " + paragraph
            else:
                merged_paragraphs.append(paragraph)
                paragraph_count += 1

        merged_sections.append((title, merged_paragraphs))

    return merged_sections, paragraph_count


folder_path = "paper_pdfs/ICLR/rejected/"

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")
        output_csv = f"paper_csvs/ICLR/rejected/{filename.replace('.pdf', '.csv')}"
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Section", "Section Appearance Order", "Paragraph"])

            try:
                doc = fitz.open(file_path)
                blocks = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    blocks += page.get_text("blocks")[:-1]

                sections, paragraph_count = extract_sections(blocks)

                app_order = 0

                for section, paragraphs in sections:
                    for paragraph in paragraphs:
                        if paragraph:
                            writer.writerow([section, app_order / paragraph_count, paragraph])
                            app_order += 1

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
