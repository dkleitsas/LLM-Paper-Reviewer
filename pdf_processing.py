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
               
                buffer = []  # Reset buffer for new section

                break  # Stop checking patterns once a match is found
       
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


"""
doc = fitz.open(file_path)
blocks = []
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    blocks += page.get_text("blocks")[:-1]

sections, paragraph_count = extract_sections(blocks)

wait = input("PRESS ENTER TO CONTINUE.")
filename = "output.csv"
with open(filename, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Paper ID", "Section", "Section Appearance Order", "Paragraph"])  # Write header
    app_order = 0

    for section, paragraphs in sections:
        for paragraph in paragraphs:
            if paragraph:  # Avoid writing empty paragraphs
                writer.writerow([section, app_order/paragraph_count, paragraph])
                app_order += 1
"""

folder_path = "paper_pdfs/"



for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")
        output_csv = f"paper_csvs/{filename.replace('.pdf', '.csv')}"
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Section", "Section Appearance Order", "Paragraph"])

            try:
                doc = fitz.open(file_path)
                blocks = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    # Maybe make it check if it's nubmer just in case?
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


"""
all_text = replace_equations(all_text)

all_text = replace_symbols(all_text)

sections = split_into_sections(all_text)

sections = categorize_sections(sections)

related_work = next((item for item in sections if item['category'] == "Related Work"), None)

with open("output.csv", "w", newline="") as file:
    fieldnames = sections[0].keys()  # Extract column names
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()  # Write headers
    writer.writerows(sections)  # Write row data

print(count_citations(related_work["content"]))
print('\n\n')

for section in sections:
    print(section)
    print("\n\n")


all_text = ""
with pdfplumber.open(file_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text(x_tolerance=2)
        if text:
            all_text += text + "\n\n"


txt_file_path = "paper_text.txt"
with open(txt_file_path, "w", encoding="utf-8") as txt_file:
    txt_file.write(all_text)


#for review in reviews:
#    if "Official_Review" in review.invitation:
#        print(review)


#for review in reviews:
    #print(review.content)
    # if "Official_Review" in review.invitation:  # Only process notes classified as reviews
    #     has_reviews = True
    #     print(f"Review: {review.content.get('review', 'No review content available')}")
    #     print(f"Rating: {review.content.get('rating', 'No rating available')}\n")"

"""
