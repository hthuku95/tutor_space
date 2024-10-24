import os
import time
from apify_client import ApifyClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import requests
import re
from django.conf import settings
from django.utils import timezone
from assignments.models import FreelancingAccount, SearchTagPairs, Assignment, OriginalPlatform

# Load environment variables
load_dotenv()

# Get API keys from environment variables
apify_api_token = os.getenv('APIFY_API_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')
stealthgpt_api_key = os.getenv('STEALTHGPT_API_KEY')

# Initialize the ApifyClient
client = ApifyClient(apify_api_token)

apify_upwork_job_auto_apply_id = os.getenv("APIFY_UPWORK_JOB_AUTO_APPLY_ID")
apify_upwork_jobs_alert_id = os.getenv("APIFY_UPWORK_JOBS_ALERT_ID")

# Initialize the LLM
model = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

# Define prompt template for proposal generation
proposal_prompt = ChatPromptTemplate.from_template(
    """Write a proposal for the following Upwork job:
    Job Title: {title}
    Job Description: {description}
    
    Your proposal should be concise, highlight relevant skills, and explain why you're a good fit for this job.
    Include a brief introduction, your relevant experience, and how you would approach the task.
    Keep the tone professional and enthusiastic.
    Do not mention specific years of experience or educational institutions.
    Focus on skills and approach relevant to the job description.
    
    End the proposal with:
    Best regards,
    {freelancer_name}"""
)

# Create proposal generation chain
proposal_chain = RunnableSequence(
    proposal_prompt,
    model,
    StrOutputParser(),
)

def search_upwork_jobs(query, limit=5):
    run_input = {
        "query": query,
        "limit": limit,
    }
    try:
        run = client.actor(apify_upwork_jobs_alert_id).call(run_input=run_input)
        result = client.dataset(run["defaultDatasetId"]).iterate_items()
        print(f"Bidding Result{result}")
        return client.dataset(run["defaultDatasetId"]).iterate_items()
    except Exception as e:
        print(f"Error in search_upwork_jobs: {str(e)}")
        return []

def extract_job_id(url):
    match = re.search(r'~([0-9a-zA-Z]+)', url)
    if match:
        return match.group(1)
    else:
        print(f"Warning: Could not extract job ID from URL: {url}")
        return None

def humanize_proposal(proposal):
    url = "https://stealthgpt.ai/api/stealthify"
    headers = {
        "api-token": stealthgpt_api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": proposal,
        "rephrase": True,
        "tone": "formal",
        "mode": "High"
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get('result', proposal)
    except requests.exceptions.RequestException as e:
        print(f"Error in humanizing proposal: {str(e)}")
        return proposal

def bid_on_job(job_url, proposal, account):
    job_id = extract_job_id(job_url)
    if not job_id:
        print("Error: Invalid job URL. Cannot proceed with bidding.")
        return False

    bid_url = f"https://www.upwork.com/ab/proposals/job/~{job_id}/apply/"
    print(f"Bidding URL: {bid_url}")

    run_input = {
        "username": account.account_gmail,
        "password": account.account_gmail_password,
        "securityQuestion": account.security_answer,
        "startUrls": [{"url": bid_url, "method": "GET"}],
        "coverLetter": proposal,
        "defaultAnswer": "I'm interested in discussing this project further.",
        "proxyConfig": {
            "useApifyProxy": True,
        },
        "debugMode": False,
        "testMode": False,
        "freelancer": f"{account.first_name} {account.last_name}",
        "agency": "",
        "autoRefill": False,
        "autoRefillAmount": "100",
        "ignoreDuplicateProposals": False
    }

    try:
        print("Calling Apify actor...")
        run = client.actor(apify_upwork_job_auto_apply_id).call(run_input=run_input)
        print(f"Actor run ID: {run['id']}")

        while True:
            run_info = client.run(run['id']).get()
            status = run_info['status']
            print(f"Current status: {status}")
            if status in ['SUCCEEDED', 'FAILED', 'TIMED-OUT', 'ABORTED']:
                break
            time.sleep(10)

        print(f"Final status: {status}")
        print(f"Exit code: {run_info.get('exitCode')}")

        if status == 'SUCCEEDED':
            print("Fetching results...")
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                print(f"Bid result: {item}")
            return True
        else:
            print(f"Actor run did not succeed. Final status: {status}")
            return False

    except Exception as e:
        print(f"An error occurred while bidding: {str(e)}")
        return False

def run_bidding_process():
    upwork_platform = OriginalPlatform.objects.get(platform_name="Upwork")
    freelancing_accounts = FreelancingAccount.objects.filter(original_platform=upwork_platform)
    search_tag_pairs = SearchTagPairs.objects.all()

    for search_tags in search_tag_pairs:
        query = f"{search_tags.tag_one} {search_tags.tag_two} {search_tags.tag_three}"
        jobs = list(search_upwork_jobs(query))

        for job in jobs:
            title = job.get('title', 'Untitled Job')
            print(f"Title: {title}")
            description = job.get('description', 'No description available')
            url = job.get('job_url') or job.get('url') or job.get('link') or 'No URL available'
            print(f"URL: {url}")
            if url == 'No URL available':
                continue

            for account in freelancing_accounts:
                print(f"Crafting Proposal...")
                initial_proposal = proposal_chain.invoke({
                    "title": title,
                    "description": description,
                    "freelancer_name": f"{account.first_name} {account.last_name}"
                })
                
                print("Humanizing proposal...")
                humanized_proposal = humanize_proposal(initial_proposal)
                print(f"Humanized Proposal: {humanized_proposal}")

                bid_success = bid_on_job(url, humanized_proposal, account)

                print(f"bid_success: {bid_success}")

                if bid_success:
                    if account.user_profile:
                        print(f"Recording the Assignment in the db")
                        # Extract completion_deadline from job data if available, or use a default value
                        completion_deadline = job.get('deadline') or (timezone.now() + timezone.timedelta(days=30))
                        try:
                            Assignment.objects.create(
                                agent=account.user_profile,
                                original_platform=upwork_platform,
                                original_account=account,
                                subject=title,
                                description=description,
                                rates=0,  # You might want to extract this from the job data if available
                                completion_deadline=completion_deadline,
                                chat_box_url=url,
                                assignment_type='P' if 'programming' in query.lower() else 'A'
                            )
                        except Exception as e:
                            print(f"Error creating Assignment: {str(e)}")
                    else:
                        print(f"No UserProfile associated with FreelancingAccount: {account.username}")

    print("Bidding process completed.")

if __name__ == "__main__":
    run_bidding_process()