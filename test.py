# Filename: rufus_client.py

# Filename: rufus_client.py

import asyncio
import logging
import json
import os
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

import ollama
from ollama import generate


STOP_WORDS = {
    'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'an', 'for', 'on', 'with', 'as', 'by', 'at', 'from',
    'this', 'that', 'these', 'those', 'are', 'be', 'or', 'if', 'we', 'you', 'your', 'our', 'but', 'not', 'can',
    'have', 'has', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'about', 'which',
    'what', 'when', 'where', 'who', 'how', 'why', 'all', 'any', 'some', 'more', 'most', 'other', 'so', 'than'
}

def clean_tokens(tokens):
    """Remove duplicates and stop words from a list of tokens."""
    tokens = set(tokens)  # Remove duplicates
    tokens -= STOP_WORDS  # Remove stop words
    return list(tokens)

def extract_keywords(user_query):
    prompt = f"""You are an intelligent keyword extraction assistant. Based on the following user query, generate a list of highly relevant keywords and phrases that cover all key aspects of the request. Include direct terms, synonyms, specific needs, and related concepts that could help address the query.
The query is: '{user_query}'. Provide the keywords in a concise, comma-separated list."""
    response = generate("llama3.1:8b", prompt, options={"temperature": 0})
    print(response['response'])

    # Extract keywords from response
    keyword_list = re.findall(r'\b[\w\s-]+\b(?=,|$)', response['response'])
    keywords = []
    for keyword in keyword_list:
        # Split phrases into words
        words = re.findall(r'\b\w+\b', keyword.strip().lower())
        keywords.extend(words)

    # Remove duplicates and common stopwords
    # stop_words = set([
    #     'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'an', 'for', 'on', 'with', 'as', 'by', 'at', 'from',
    #     'this', 'that', 'these', 'those', 'are', 'be', 'or', 'if', 'we', 'you', 'your', 'our', 'but', 'not', 'can',
    #     'have', 'has', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'about', 'which',
    #     'what', 'when', 'where', 'who', 'how', 'why', 'all', 'any', 'some', 'more', 'most', 'other', 'so', 'than'
    # ])
    # keywords = [word for word in set(keywords) if word not in stop_words]
    keywords = clean_tokens(keywords)


    return keywords

class RufusClient:
    def __init__(self, max_concurrency=5, max_pages=100, depth_threshold=1):

        self.max_concurrency = max_concurrency
        self.max_pages = max_pages
        self.depth_threshold = depth_threshold
        self.visited_urls = []
        self.to_visit = asyncio.Queue()
        self.results = []
        self.logger = self._setup_logger()
        self.playwright = None
        self.browser = None
        self.context = None
        self.instructions = ""
        self.base_domain = ""
        self.keywords = []
        self.start_url = None

    def _setup_logger(self):
        logger = logging.getLogger('RufusClient')
        logger.setLevel(logging.INFO)  # Set to logging.DEBUG for more details
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def scrape(self, url, instructions):

        self.instructions = instructions
        self.base_domain = urlparse(url).netloc
        self.start_url = url  # Set the start URL

        # Extract keywords from instructions using Ollama
        self.keywords = extract_keywords(instructions)
        self.logger.info(f"Extracted Keywords: {self.keywords}")

        await self.to_visit.put((url, 0))  # Start at depth 0

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()

        workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.max_concurrency)
        ]

        await self.to_visit.join()

        for worker in workers:
            worker.cancel()

        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()

        return self.results

    async def worker(self):
   
        while True:
            url, depth = await self.to_visit.get()
            if url not in self.visited_urls and len(self.visited_urls) < self.max_pages:
                self.visited_urls.append(url)
                try:
                    await self.process_page(url, depth)
                except Exception as e:
                    self.logger.error(f"Error processing {url}: {e}")
            self.to_visit.task_done()

    async def process_page(self, url, depth):

        self.logger.info(f"Processing URL: {url} at depth {depth}")
        page = await self.context.new_page()
        try:
            await page.goto(url, timeout=30000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

            # Decide whether to apply content-based filtering based on depth
            if depth <= self.depth_threshold or self.is_potentially_relevant(text):
                data = self.extract_information(text, url)
                if data['content']:
                    self.results.append(data)
            else:
                self.logger.info(f"Skipping irrelevant page based on keywords: {url}")

            # Enqueue links with increased depth
            await self.enqueue_links(soup, url, depth + 1)
        except Exception as e:
            self.logger.error(f"Failed to process {url}: {e}")
        finally:
            await page.close()

    async def enqueue_links(self, soup, base_url, depth):

        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            link_text = link_tag.get_text(separator=' ', strip=True).lower()
            next_url = urljoin(base_url, href)
            if self.should_visit_url(next_url):
                await self.to_visit.put((next_url, depth))

    def should_visit_url(self, url):

        parsed_url = urlparse(url)
        if parsed_url.netloc != self.base_domain:
            return False

        if url in self.visited_urls:
            return False

        # Exclude certain paths and file types
        excluded_paths = ['/login', '/signup', '/register', '/cart', '/checkout']
        if any(path in parsed_url.path.lower() for path in excluded_paths):
            return False

        excluded_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.rar', '.css', '.js']
        if any(parsed_url.path.lower().endswith(ext) for ext in excluded_extensions):
            return False

        return True  # Do not apply keyword filtering on URLs and link text

    def is_potentially_relevant(self, text):
  
        tokens = re.findall(r'\b\w+\b', text.lower())

        # stop_words = set([
        #     'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'an', 'for', 'on', 'with', 'as', 'by', 'at', 'from',
        #     'this', 'that', 'these', 'those', 'are', 'be', 'or', 'if', 'we', 'you', 'your', 'our', 'but', 'not', 'can',
        #     'have', 'has', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'about', 'which',
        #     'what', 'when', 'where', 'who', 'how', 'why', 'all', 'any', 'some', 'more', 'most', 'other', 'so', 'than'
        # ])
        # tokens = [token for token in tokens if token not in stop_words]
        tokens = clean_tokens(tokens)


        matches = set(self.keywords) & set(tokens)
        if matches:
            self.logger.debug(f"Keywords matched: {matches}")
            return True
        return False

    def extract_information(self, text, url):

        sentences = re.split(r'(?<=[.!?]) +', text)
        relevant_sentences = []

        for sentence in sentences:
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            tokens = clean_tokens(tokens)

            # tokens = [token for token in tokens if token not in [
            #     'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'an', 'for', 'on', 'with', 'as', 'by', 'at', 'from',
            #     'this', 'that', 'these', 'those', 'are', 'be', 'or', 'if', 'we', 'you', 'your', 'our', 'but', 'not', 'can',
            #     'have', 'has', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'about', 'which',
            #     'what', 'when', 'where', 'who', 'how', 'why', 'all', 'any', 'some', 'more', 'most', 'other', 'so', 'than'
            # ]]
            if set(self.keywords) & set(tokens):
                relevant_sentences.append(sentence.strip())

        return {
            "url": url,
            "content": ' '.join(relevant_sentences)
        }

# Example usage
if __name__ == "__main__":
    import sys

    async def main():
        client = RufusClient(max_concurrency=5, max_pages=100, depth_threshold=1)
        instructions = "i need financial assiatnce from government"
        documents = await client.scrape("https://governor.mo.gov/", instructions)
        print(json.dumps(documents, indent=2))

    asyncio.run(main())



#tested linls

#https://www.sf.gov/
#https://governor.mo.gov/

