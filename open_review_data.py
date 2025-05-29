import openreview
import requests

from openreview.api import OpenReviewClient


def save_paper(pdf_url, filename):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)


def get_convention_papers(convention_id, client, paper_limit=10, file_path="paper_pdfs", use_api2=False):

    papers = client.get_notes(invitation=convention_id, limit=paper_limit, offset=1000)

    papers_downloaded = 0
    accepted_count = 0
    rejected_count = 0

    for i, paper in enumerate(papers):
        if use_api2:
            title = paper.content.get('title', f'No title ({i})').get('value', None)
            pdf_id = paper.content.get('pdf', None).get('value', None)
        else:
            title = paper.content.get('title', f'No title ({i})')
            pdf_id = paper.content.get('pdf', None)

        if not pdf_id:
            print(f"Paper '{title}' has no PDF path.")
            continue

        pdf_url = f"https://openreview.net{pdf_id}"


        decision_notes = client.get_notes(
            forum=paper.forum,
            invitation=f'{convention_id.replace("/-/Blind_Submission", "")}/Paper{paper.number}/-/Decision'
            )
        
        if decision_notes:
            # decision_value = decision_notes[0].content.get('decision', {}).get('value', None) Depends on API version idc enough to make it dynamic
            decision_value = decision_notes[0].content.get('decision', {})
        else:
            decision_value = None

        decision_class = "accepted" if "accept" in decision_value.lower() else "rejected" if "reject" in decision_value.lower() else "other"



        if decision_class == "other" or (decision_class == "rejected" and (accepted_count - rejected_count < 50)):
            print(f"Decision for paper '{title}' is not clear: {decision_value}. Skipping. Or too many accepted papers.")
            continue
        if decision_class == "accepted":
            accepted_count += 1
        elif decision_class == "rejected":
            rejected_count += 1



        download_path = f"{file_path}/{decision_class}/{paper.id}.pdf"
        
        print(f"Processing: {title}")


        save_paper(pdf_url, download_path)
        papers_downloaded += 1

    print(papers_downloaded)




def main():
    convention_id = "ICLR.cc/2022/Conference/-/Blind_Submission"              # ICLR 2023 10 papers
    # convention_id = "NeurIPS.cc/2022/Conference/-/Blind_Submission"           # NeurIPS 2022 10 papers
    # convention_id = "Computo/-/Submission"                                    # Computo 2024 6 papers   
    # convention_id = "logconference.io/LOG/2022/Conference/-/Blind_Submission" # LOG 2022 10 papers
    # convention_id = "XJTU.edu.cn/2024/CSUC/-/Submission"                      # ΧJTU CSUC 2024 10 papers        
    # convention_id = "ICOMP.cc/2024/Conference/-/Submission"                   # ICOMP 2024 10 papers   
    # convention_id = 'MIDL.io/2024/Conference/-/Submission'
    paper_limit = 1000
    file_path = "paper_pdfs/ICLR"
        
    # Some older venues only available with old API
    # Some newer venues only with new API 
    # Stupid stupid stupid stupid stupid

    # Utility func to check what the invitation looks like
    try:
        client = OpenReviewClient(baseurl='https://api2.openreview.net')
        note = client.get_note('youe3QQepVB')
        use_api2 = True
        print(note.invitations)
    except Exception as e:
        print("Not available with API2. Trying API1.")
        client = openreview.Client(baseurl='https://api.openreview.net')
        note = client.get_note('youe3QQepVB')
        use_api2 = False
        print(note.invitation)


    get_convention_papers(convention_id, client, paper_limit, file_path, use_api2=use_api2)


if __name__ == "__main__":
    main()


