# content_scraper.py
# Created: 2025-02-03 12:03:46 UTC
# Last modified: 2025-02-03 12:03:46 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import re
import json
import time
import logging
import aiohttp
import asyncio
import hashlib
import threading
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from config import CONFIG
from exceptions import (
    ScrapingError, 
    ContentExtractionError, 
    NetworkError, 
    ParsingError,
    ResourceLimitError
)
from utils import MemoryManager, FileSystemWatcher, DownloadManager

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.RotatingFileHandler(
            'scraper.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContentScraper:
    """Enhanced content scraper with advanced features and optimizations.
    
    Features:
    - Async content downloading
    - Smart rate limiting
    - Automatic retry mechanism
    - Resource management
    - Content validation
    - Memory optimization
    - Caching support
    """
    
    def __init__(
        self,
        base_url: str,
        output_dir: Union[str, Path],
        max_retries: int = CONFIG['scraping']['max_retries'],
        timeout: int = CONFIG['scraping']['timeout'],
        cache_enabled: bool = True
    ):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        
        # Initialize components
        self._setup_selenium()
        self._setup_async_session()
        self._setup_memory_manager()
        self._setup_file_watcher()
        self._setup_download_manager()
        
        # Statistics and monitoring
        self.stats = {
            'start_time': datetime.utcnow(),
            'pages_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_bytes_downloaded': 0,
            'errors': []
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _setup_selenium(self) -> None:
        """Initialize Selenium with optimized settings."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-logging')
            chrome_options.add_argument('--log-level=3')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Selenium initialized successfully")
        except Exception as e:
            logger.error(f"Selenium initialization failed: {str(e)}")
            raise ScrapingError("Failed to initialize Selenium", original_error=e)

    async def _setup_async_session(self) -> None:
        """Initialize async session with optimal settings."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)

    def _setup_memory_manager(self) -> None:
        """Initialize memory management."""
        self.memory_manager = MemoryManager(
            threshold=CONFIG['resources']['max_memory_percent'],
            leak_detection=True
        )

    def _setup_file_watcher(self) -> None:
        """Initialize file system monitoring."""
        self.file_watcher = FileSystemWatcher(
            str(self.output_dir),
            patterns=["*.html", "*.json"],
            recursive=True
        )

    def _setup_download_manager(self) -> None:
        """Initialize download management."""
        self.download_manager = DownloadManager(
            max_concurrent=CONFIG['download']['max_concurrent'],
            rate_limit=CONFIG['download']['rate_limit'],
            cache_size=CONFIG['download']['cache_size']
        )

    async def extract_content(self, url: str) -> Dict[str, Any]:
        """Extract content from URL with enhanced error handling and validation."""
        try:
            async with self._lock:
                self.stats['pages_processed'] += 1
                
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
                
            # Check cache if enabled
            if self.cache_enabled:
                cached_content = await self._get_cached_content(url)
                if cached_content:
                    return cached_content
            
            # Fetch and parse content
            html = await self._fetch_page(url)
            content = await self._parse_content(html)
            
            # Validate extracted content
            if not self._validate_content(content):
                raise ContentExtractionError(f"Invalid content structure for {url}")
            
            # Update statistics
            async with self._lock:
                self.stats['successful_extractions'] += 1
                self.stats['total_bytes_downloaded'] += len(html)
            
            return content
            
        except Exception as e:
            async with self._lock:
                self.stats['failed_extractions'] += 1
                self.stats['errors'].append(str(e))
            raise ContentExtractionError(f"Content extraction failed for {url}", original_error=e)

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and allowed domains."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def _get_cached_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached content if available."""
        if not self.cache_enabled:
            return None
            
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.output_dir / 'cache' / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    content = json.loads(await f.read())
                logger.info(f"Cache hit for {url}")
                return content
            except Exception as e:
                logger.warning(f"Cache read failed for {url}: {str(e)}")
                
        return None

    def cleanup(self) -> None:
        """Cleanup resources and generate final report."""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
            
            if hasattr(self, 'session'):
                asyncio.run(self.session.close())
            
            # Generate final report
            self._generate_report()
            
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

    def _generate_report(self) -> None:
        """Generate detailed scraping report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'duration': str(datetime.utcnow() - self.stats['start_time']),
            'stats': self.stats,
            'config': {
                'base_url': self.base_url,
                'max_retries': self.max_retries,
                'timeout': self.timeout,
                'cache_enabled': self.cache_enabled
            }
        }
        
        report_file = self.output_dir / f"scraping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Scraping report generated: {report_file}")

# Last modified: 2025-02-03 12:03:46 UTC
# End of content_scraper.py