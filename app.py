
"""
Company Website Summarizer
Run with: python app.py
"""

import os
import requests
import gradio as gr

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from requests.exceptions import (
    RequestException,
    Timeout,
    HTTPError,
)


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Check your .env file."
    )

print("[OK] API key loaded")

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# 2. Website settings
# --------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

JUNK_TAGS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
]

MAX_CHARS = 8000


# --------------------------------------------------
# 3. Fetch and clean website
# --------------------------------------------------

def fetch_website(url):
    """Fetch website content and extract readable text."""

    url = url.strip()

    if not url:
        raise ValueError("Please enter a website URL.")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )

        response.raise_for_status()

    except Timeout:
        raise RuntimeError(
            "The website took too long to respond."
        )

    except HTTPError as error:
        raise RuntimeError(
            f"Website returned an HTTP error: {error}"
        )

    except RequestException as error:
        raise RuntimeError(
            f"Could not access the website: {error}"
        )

    # ----------------------------------------------
    # Handle website encoding safely
    # ----------------------------------------------

    encoding = response.apparent_encoding

    if encoding:
        response.encoding = encoding
    else:
        response.encoding = "utf-8"

    # Decode content safely to avoid Unicode errors
    html = response.content.decode(
        response.encoding or "utf-8",
        errors="replace",
    )

    # ----------------------------------------------
    # Parse HTML
    # ----------------------------------------------

    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(
        strip=True
    ) if soup.title else "Untitled Website"

    # Remove unnecessary HTML elements
    for tag in soup(JUNK_TAGS):
        tag.decompose()

    # Extract readable text
    text = soup.get_text(
        separator=" ",
        strip=True,
    )

    # Normalize whitespace
    text = " ".join(text.split())

    # Remove invalid characters safely
    text = text.encode(
        "utf-8",
        errors="replace",
    ).decode(
        "utf-8",
        errors="replace",
    )

    # Limit text length
    text = text[:MAX_CHARS]

    return title, url, text


# --------------------------------------------------
# 4. OpenAI instructions
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a professional business analyst.

Analyze the website content provided by the user.

Create a clear and useful business summary using
the following headings:

## What the company does

## Products and services

## Who their customers are

## News and announcements

## How they present themselves

## Notes for the owner

Important rules:

1. Use only information available in the website text.
2. Do not invent facts.
3. If information is missing, clearly mention that.
4. Use simple and professional language.
5. Keep the summary structured and readable.
6. Do not claim that you visited pages that were
   not included in the provided text.
"""


def build_messages(owner, title, url, page_text):
    """Build messages for the OpenAI API."""

    user_prompt = f"""
Company owner name: {owner}

Website title: {title}

Website URL: {url}

Website content:
{page_text}

Prepare a structured business summary.
"""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


# --------------------------------------------------
# 5. Generate summary with streaming
# --------------------------------------------------

def summarize(owner, url, model):
    """Generate a streamed website summary."""

    owner = owner.strip()
    url = url.strip()

    if not owner:
        yield "Please enter the company owner's name."
        return

    if not url:
        yield "Please enter a website URL."
        return

    yield "🔍 Fetching website content...\n"

    try:
        title, final_url, page_text = fetch_website(url)

    except Exception as error:
        yield f"❌ Error: {error}"
        return

    if len(page_text.strip()) < 50:
        yield (
            "❌ Very little readable text was found on this website. "
            "The website may use JavaScript or block automated access."
        )
        return

    yield (
        f"✅ Website fetched successfully.\n\n"
        f"**Website:** {final_url}\n\n"
        f"**Generating summary...**\n\n"
    )

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=build_messages(
                owner,
                title,
                final_url,
                page_text,
            ),
            stream=True,
        )

        complete_answer = ""

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                complete_answer += delta.content

                yield (
                    f"**Website:** {final_url}\n\n"
                    + complete_answer
                )

    except Exception as error:
        yield (
            "❌ OpenAI API error:\n\n"
            f"{error}"
        )


# --------------------------------------------------
# 6. Gradio interface
# --------------------------------------------------

with gr.Blocks(
    title="Company Website Summarizer"
) as app:

    gr.Markdown(
        """
# 🌐 Company Website Summarizer

Enter a company owner's name and website URL
to generate a structured business summary.
"""
    )

    with gr.Row():

        owner_input = gr.Textbox(
            label="Company Owner Name",
            placeholder="Enter owner's name",
        )

        url_input = gr.Textbox(
            label="Company Website URL",
            placeholder="https://example.com",
        )

    model_input = gr.Dropdown(
        choices=[
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4o-mini",
        ],
        value="gpt-4.1-mini",
        label="Select OpenAI Model",
    )

    summarize_button = gr.Button(
        "Generate Summary",
        variant="primary",
    )

    output = gr.Markdown(
        label="Business Summary"
    )

    summarize_button.click(
        fn=summarize,
        inputs=[
            owner_input,
            url_input,
            model_input,
        ],
        outputs=output,
    )

    url_input.submit(
        fn=summarize,
        inputs=[
            owner_input,
            url_input,
            model_input,
        ],
        outputs=output,
    )

    gr.Markdown(
        """
**Tip:** Some websites block automated requests
or load their content using JavaScript.
"""
    )


# --------------------------------------------------
# 7. Launch application
# --------------------------------------------------

if __name__ == "__main__":
    app.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7878)),
    )