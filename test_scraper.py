# test_scraper.py
# Created: 2025-02-03 14:19:00 UTC
# Last modified: 2025-02-03 14:19:00 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import json
import pytest
import asyncio
import aiohttp
import logging
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from web_scraper import WebScraper
from content_processor import ContentProcessor
from single_file import SingleFileProcessor
from config import CONFIG
from exceptions import ScrapingError, ProcessingError

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test data directory setup
TEST_DIR = Path(__file__).parent / 'test_data'
TEST_DIR.mkdir(exist_ok=True)

# Sample test data
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Test Content</h1>
    <p>This is a test paragraph.</p>
    <a href="http://example.com">Test Link</a>
    <img src="test.jpg" alt="Test Image">
</body>
</html>
"""

SAMPLE_URLS = [
    'http://example.com/test1',
    'http://example.com/test2',
    'http://example.com/test3'
]

@pytest.fixture
def web_scraper():
    """Fixture for WebScraper instance"""
    scraper = WebScraper()
    yield scraper
    asyncio.run(scraper.close())

@pytest.fixture
def content_processor():
    """Fixture for ContentProcessor instance"""
    return ContentProcessor(
        input_dir=TEST_DIR,
        output_dir=TEST_DIR
    )

@pytest.fixture
def single_file_processor():
    """Fixture for SingleFileProcessor instance"""
    return SingleFileProcessor(
        output_dir=TEST_DIR
    )

@pytest.fixture
def sample_response():
    """Fixture for sample aiohttp response"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value=SAMPLE_HTML)
    mock_response.headers = {'content-type': 'text/html'}
    return mock_response

class TestWebScraper:
    """Test suite for WebScraper class"""

    @pytest.mark.asyncio
    async def test_init(self, web_scraper):
        """Test WebScraper initialization"""
        assert web_scraper is not None
        assert hasattr(web_scraper, 'session')
        assert web_scraper.rate_limit == CONFIG['RATE_LIMIT']

    @pytest.mark.asyncio
    async def test_process_single_url(self, web_scraper, sample_response):
        """Test processing single URL"""
        with patch('aiohttp.ClientSession.get', return_value=sample_response):
            result = await web_scraper.process_single_url('http://example.com')
            assert result is not None
            assert 'url' in result
            assert 'html' in result
            assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_process_urls_batch(self, web_scraper, sample_response):
        """Test processing multiple URLs in batch"""
        with patch('aiohttp.ClientSession.get', return_value=sample_response):
            results = await web_scraper.process_urls(SAMPLE_URLS[:2])
            assert len(results) == 2
            assert all('url' in result for result in results)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, web_scraper, sample_response):
        """Test rate limiting functionality"""
        start_time = datetime.now()
        with patch('aiohttp.ClientSession.get', return_value=sample_response):
            await web_scraper.process_urls(SAMPLE_URLS)
        duration = (datetime.now() - start_time).total_seconds()
        assert duration >= (len(SAMPLE_URLS) - 1) * web_scraper.rate_limit

    @pytest.mark.asyncio
    async def test_error_handling(self, web_scraper):
        """Test error handling for failed requests"""
        mock_error_response = AsyncMock()
        mock_error_response.status = 404
        
        with patch('aiohttp.ClientSession.get', return_value=mock_error_response):
            with pytest.raises(ScrapingError):
                await web_scraper.process_single_url('http://example.com/notfound')

class TestContentProcessor:
    """Test suite for ContentProcessor class"""

    def test_init(self, content_processor):
        """Test ContentProcessor initialization"""
        assert content_processor is not None
        assert content_processor.input_dir == TEST_DIR
        assert content_processor.output_dir == TEST_DIR

    @pytest.mark.asyncio
    async def test_process_content(self, content_processor):
        """Test content processing"""
        test_content = {
            'url': 'http://example.com',
            'html': SAMPLE_HTML,
            'timestamp': datetime.now().isoformat()
        }
        result = await content_processor.process_content(test_content)
        assert result is not None
        assert 'processed_html' in result
        assert 'metadata' in result

    @pytest.mark.asyncio
    async def test_batch_processing(self, content_processor):
        """Test batch content processing"""
        test_contents = [
            {
                'url': url,
                'html': SAMPLE_HTML,
                'timestamp': datetime.now().isoformat()
            }
            for url in SAMPLE_URLS
        ]
        results = await content_processor.process_batch(test_contents)
        assert len(results) == len(test_contents)
        assert all('processed_html' in result for result in results)

class TestSingleFileProcessor:
    """Test suite for SingleFileProcessor class"""

    def test_init(self, single_file_processor):
        """Test SingleFileProcessor initialization"""
        assert single_file_processor is not None
        assert single_file_processor.output_dir == TEST_DIR

    @pytest.mark.asyncio
    async def test_process_html(self, single_file_processor):
        """Test HTML processing to single file"""
        result = await single_file_processor.process_html(
            SAMPLE_HTML,
            'http://example.com'
        )
        assert result is not None
        assert 'data:' in result

    def test_resource_extraction(self, single_file_processor):
        """Test resource extraction from HTML"""
        resources = single_file_processor._extract_resources(SAMPLE_HTML)
        assert len(resources) > 0
        assert any(r['type'] == 'image' for r in resources)
        assert any(r['type'] == 'link' for r in resources)

class TestIntegration:
    """Integration tests for the complete scraping pipeline"""

    @pytest.mark.asyncio
    async def test_complete_pipeline(
        self,
        web_scraper,
        content_processor,
        single_file_processor,
        sample_response
    ):
        """Test complete scraping pipeline"""
        # Mock web scraping
        with patch('aiohttp.ClientSession.get', return_value=sample_response):
            # Scrape URLs
            scraped_results = await web_scraper.process_urls(SAMPLE_URLS[:1])
            assert len(scraped_results) == 1
            
            # Process content
            processed_results = await content_processor.process_batch(scraped_results)
            assert len(processed_results) == 1
            
            # Create single file
            single_file = await single_file_processor.process_html(
                processed_results[0]['processed_html'],
                processed_results[0]['url']
            )
            assert single_file is not None

def test_cleanup():
    """Clean up test data directory"""
    try:
        for file in TEST_DIR.glob('*'):
            file.unlink()
        TEST_DIR.rmdir()
    except Exception as e:
        logger.warning(f"Cleanup failed: {str(e)}")

# Last modified: 2025-02-03 14:19:00 UTC
# End of test_scraper.py